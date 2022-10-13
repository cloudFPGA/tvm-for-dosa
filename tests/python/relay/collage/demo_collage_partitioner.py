# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Compares Collage with various other baselines."""

# CAUTION: Requires some changes in python/tvm/autotvm/task/dispatcher.py
# so that AutoTVM tuning records can be cached between runs and between
# models. See https://github.com/mbs-octoml/mbs-tvm/tree/mbs-collage-hacks.

import tvm
import logging
import tempfile
import os
import shutil

import menangerie

# The following are necessary to force global functions or pattern tables to be registered
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass import num_cutlass_partitions
from tvm.relay.op.contrib.cublas import partition_for_cublas
from tvm.relay.op.contrib.cudnn import partition_for_cudnn

logging.basicConfig(level=logging.INFO)


########### Configuration ###########

###
### Rename to match your hardware, eg ..._vt100...
###
TUNING_LOG = "/home/mbs/collage_autotvm_rtx3070.tuninglog"

###
### If true, runs final model under nvprof
###
PROFILE = True

###
### If true, run all models
###
ALL_MODELS = False

###
### If true, run all configurations
###
ALL_CONFIGS = False

###
### How aggressively to look for candidates?
###
TVM_MAX_DEPTH = 8
BYOC_MAX_DEPTH = 8

###
### AutoTVM tuning parameters.
###
AUTOTVM_NUM_TRIALS = 2000
AUTOTVM_EARLY_STOPPING = 600
TIMEOUT = 10
MEASURE_NUMBER = tvm.relay.collage.MEASURE_NUMBER
MEASURE_REPEAT = tvm.relay.collage.MEASURE_REPEAT
WARMUP_MIN_REPEAT_MS = tvm.relay.collage.WARMUP_MIN_REPEAT_MS

HOST = tvm.target.Target("llvm")
CUDA = tvm.target.Target("cuda", HOST)

########### Runtime ###########

# Code to run a model. The actual call to 'run' is appended at compile time.
# We invoke the model as a sub-process so that we can wrap profiling tools around it.
runner_template = f"""
import tvm
import tvm.runtime.vm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

MEASURE_NUMBER = {MEASURE_NUMBER}
MEASURE_REPEAT = {MEASURE_REPEAT}
WARMUP_MIN_REPEAT_MS = {WARMUP_MIN_REPEAT_MS}

def arg_for(shape, dtype, device):
    return tvm.nd.array(
        np.random.rand(*shape).astype(dtype), device=device)

def vm_estimate_seconds(device, vm, args):
    vm.benchmark(device, repeat=1, number=1, min_repeat_ms=WARMUP_MIN_REPEAT_MS, **args)
    return vm.benchmark(device, repeat=MEASURE_REPEAT, number=MEASURE_NUMBER, min_repeat_ms=0,
                        **args)


def run(label, name, device, lib_path, code_path, input_shapes, input_dtypes):
    logging.info(f"Loading compiled code for {{name}} generated by {{label}} from {{lib_path}} and {{code_path}}...")
    loaded_lib = tvm.runtime.load_module(lib_path)
    loaded_code = bytearray(open(code_path, "rb").read())
    loaded_exe = tvm.runtime.vm.Executable.load_exec(loaded_code, loaded_lib)
    vm = tvm.runtime.vm.VirtualMachine(loaded_exe, device)
    args = {{
        input_name: arg_for(input_shapes[input_name], input_dtypes[input_name], device)
        for input_name in input_shapes.keys()
    }}
    logging.info(f"Benchmarking for {{name}} generated by {{label}}...")
    profile = vm_estimate_seconds(device, vm, args) 
    logging.info(f"Benchmarked for {{name}} generated by {{label}}: {{profile}}")
    logging.info(f"RESULT: {{label}} | {{name}} | {{profile.median * 1e3}}ms")

if __name__ == "__main__":
"""

########### AutoTVM tuning helpers ###########


def extract_autotvm_tasks(mod, target):
    """Returns TVM kernels to tune for mod and target."""
    return tvm.autotvm.task.extract_from_program(mod, target=target, params=None)


def optional_tuning_records(log_filename):
    """Returns existing tuning records, if any."""
    if log_filename == "" or not os.path.exists(log_filename):
        return tvm.autotvm.task.FallbackContext()
    else:
        return tvm.autotvm.task.ApplyHistoryBest(log_filename)


def is_already_tuned(task, log_filename):
    """Returns True if we already have a tuning record for task in turning logs in log_filename"""
    if not os.path.exists(log_filename):
        return False

    dispatch_context = tvm.autotvm.task.ApplyHistoryBest(log_filename)
    return dispatch_context.contains(task.target, task.workload)


def tune_autotvm_tasks(tasks, log_filename):
    """Appends to log_filename the best strategies for tasks"""
    if len(tasks) == 0:
        return

    measure_option = tvm.autotvm.measure_option(
        builder=tvm.autotvm.LocalBuilder(timeout=TIMEOUT),
        runner=tvm.autotvm.LocalRunner(
            number=MEASURE_NUMBER, repeat=MEASURE_REPEAT, timeout=TIMEOUT, min_repeat_ms=0
        ),
    )

    logging.info(
        f"Using autotvm tuning for {len(tasks)} tasks with {AUTOTVM_NUM_TRIALS} trials, logging to {log_filename}"
    )

    # create tmp log file, starting with contents from existing log file
    tmp_log_filename = log_filename + ".tmp"
    if os.path.exists(tmp_log_filename):
        os.remove(tmp_log_filename)
    if os.path.exists(log_filename):
        logging.info(f"Copying existing log {log_filename} to {tmp_log_filename}")
        shutil.copy(log_filename, tmp_log_filename)

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        logging.info(f"Considering task {task.name} {prefix}")
        if is_already_tuned(task, tmp_log_filename):
            logging.info(f"Re-using existing record for {task.name}")
            continue

        logging.info(f"Using autotvm to tune {task.name}")
        tuner_obj = tvm.autotvm.tuner.XGBTuner(task, loss_type="rank")
        if os.path.exists(tmp_log_filename):
            tuner_obj.load_history(tvm.autotvm.record.load_from_file(tmp_log_filename))

        # do tuning
        n_trial = min(AUTOTVM_NUM_TRIALS, len(task.config_space))
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=AUTOTVM_EARLY_STOPPING,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.progress_bar(n_trial, prefix=prefix),
                tvm.autotvm.callback.log_to_file(tmp_log_filename),
            ],
        )

    # pick best records and copy back to main log file
    tvm.autotvm.record.pick_best(tmp_log_filename, log_filename)
    os.remove(tmp_log_filename)

    logging.info("Done with autotvm tuning")


def autotvm_tune_module(mod, target, log_filename):
    if log_filename == "":
        logging.info("Not tuning with autotvm since disabled")
        return
    # Extract and tune any TVM kernels. BYOC partitions will have no tasks extracted.
    logging.info("Extracting tasks from overall module")
    tasks = extract_autotvm_tasks(mod, target)
    logging.info(f"Auto-tuning {len(tasks)} tasks from overall module")
    tune_autotvm_tasks(tasks, log_filename)


########### Drivers ###########


def compile_and_benchmark(label, model, targets, dev, tmp_dir):
    """Compile model for target and run it with profiling."""
    logging.info(f"Compiling {model['name']} using {label} with {targets}...")
    exe = tvm.relay.vm.compile(model["mod"], target=targets, params=model["params"])
    lib_path = os.path.join(tmp_dir, "lib.so")
    code_path = os.path.join(tmp_dir, "code.ro")
    code, lib = exe.save()
    logging.info(f"Saving VM code to {code_path}...")
    with open(code_path, "wb") as fo:
        fo.write(code)
    logging.info(f"Exporting library to {lib_path}...")
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    runner = f"{runner_template}    run('{label}', '{model['name']}', tvm.device({dev.device_type}), '{lib_path}', '{code_path}', {model['input_shapes']}, {model['input_dtypes']})\n"
    runner_path = os.path.join(tmp_dir, "runner.py")
    logging.info(f"Saving runner to {runner_path}...")
    with open(runner_path, "w") as fo:
        fo.write(runner)

    logging.info(f"Invoking runner...")
    if PROFILE:
        profile_path = os.path.join(tmp_dir, "profile.txt")
        os.system(f"nsys nvprof -o {profile_path} python3 {runner_path}")
    else:
        os.system(f"python3 {runner_path}")


def collage(model):
    """Run the Collage partitioner for a set of CUDA-related targets and profile the result"""
    logging.info(f"collage | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    autotvm_tune_module(model["mod"], CUDA, TUNING_LOG)
    with optional_tuning_records(TUNING_LOG):
        targets = []
        targets.append(CUDA)
        use_fp16 = model["main_dtype"] == "float16"
        targets.append(
            tvm.target.Target(f"tensorrt -use_implicit_batch=False -use_fp16={use_fp16}", HOST)
        )
        tmp_dir = tempfile.mkdtemp()
        targets.append(tvm.target.Target(f"cutlass -tmp_dir={tmp_dir}", HOST))
        targets.append(tvm.target.Target("cublas", HOST))
        targets.append(tvm.target.Target("cudnn", HOST))
        config = {
            "relay.collage.tvm_max_depth": TVM_MAX_DEPTH,
            "relay.collage.byoc_max_depth": BYOC_MAX_DEPTH,
        }
        logging.info(f"Using PassContext(config={config}")
        ctxt = tvm.transform.PassContext(config=config)
        config = tvm.target.make_compilation_config(ctxt, targets)
        with ctxt:
            mod = model["mod"]
            mod = tvm.relay.transform.CapturePostDfsIndexInSpans()(mod)
            logging.info("-------------- BEGIN INDEXED --------------")
            logging.info(mod)
            logging.info("-------------- END INDEXED ----------------")
            mod = tvm.relay.transform.CollagePartition(config)(mod)
            partitioned_model = model.copy()
            partitioned_model["mod"] = mod
            logging.info("-------------- BEGIN PARTITIONED --------------")
            logging.info(partitioned_model["mod"])
            logging.info("-------------- END PARTITIONED ----------------")
            dev = tvm.device(CUDA.get_target_device_type())
            compile_and_benchmark("collage", partitioned_model, targets, dev, tmp_dir)


def just_tensorrt(model):
    """Run partition_for_tensorrt, complete the compilation with TVM, and profile the result."""
    logging.info(f"just_tensorrt | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    tmp_dir = tempfile.mkdtemp()
    autotvm_tune_module(model["mod"], CUDA, TUNING_LOG)
    with optional_tuning_records(TUNING_LOG):
        logging.info("Partitioning for TensorRT...")
        use_fp16 = model["main_dtype"] == "float16"
        trt_target = tvm.target.Target(
            f"tensorrt -use_implicit_batch=False -use_fp16={use_fp16}", HOST
        )
        mod = tvm.relay.op.contrib.partition_for_tensorrt(
            mod=model["mod"], params=model["params"], target=trt_target
        )
        partitioned_model = model.copy()
        partitioned_model["mod"] = mod
        logging.info("-------------- BEGIN PARTITIONED --------------")
        logging.info(partitioned_model["mod"])
        logging.info("-------------- END PARTITIONED ----------------")
        targets = []
        targets.append(CUDA)
        targets.append(trt_target)
        dev = tvm.device(CUDA.get_target_device_type())
        compile_and_benchmark("just_tensorrt", partitioned_model, targets, dev, tmp_dir)


def just_cutlass(model):
    """Run partition_for_cutlass, complete the compilation with TVM, and profile the result."""
    logging.info(f"just_cutlass | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    tmp_dir = tempfile.mkdtemp()
    autotvm_tune_module(model["mod"], CUDA, TUNING_LOG)
    with optional_tuning_records(TUNING_LOG):
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            logging.info("Partitioning for CUTLASS...")
            mod = tvm.relay.op.contrib.partition_for_cutlass(model["mod"], model["params"])
            partitioned_model = model.copy()
            partitioned_model["mod"] = mod
            logging.info("-------------- BEGIN PARTITIONED --------------")
            logging.info(partitioned_model["mod"])
            logging.info("-------------- END PARTITIONED ----------------")
            targets = []
            targets.append(CUDA)
            targets.append(tvm.target.Target(f"cutlass -tmp_dir={tmp_dir}", HOST))
            dev = tvm.device(CUDA.get_target_device_type())
            compile_and_benchmark("just_cutlass", partitioned_model, targets, dev, tmp_dir)


def just_tvm(model):
    """Compile and profile using vanilla TVM."""
    logging.info(f"just_tvm | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    tmp_dir = tempfile.mkdtemp()
    autotvm_tune_module(model["mod"], CUDA, TUNING_LOG)
    with optional_tuning_records(TUNING_LOG):
        dev = tvm.device(CUDA.get_target_device_type())
        compile_and_benchmark("just_tvm", model, CUDA, dev, tmp_dir)


def tvm_with_libs(model):
    """As for just_tvm, but use the existing -libs mechanism to enable standard CUDA libs."""
    logging.info(f"tvm_with_libs | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    tmp_dir = tempfile.mkdtemp()
    cuda_target = tvm.target.Target("cuda -libs=cudnn,cublas", HOST)
    autotvm_tune_module(model["mod"], cuda_target, TUNING_LOG)
    with optional_tuning_records(TUNING_LOG):
        dev = tvm.device(cuda_target.get_target_device_type())
        compile_and_benchmark("tvm_with_libs", model, cuda_target, dev, tmp_dir)


########### Runners ###########


def run_all():
    """Run the whole test suite."""
    make_models = []
    make_models.append(menangerie.resnext50_32x4d)
    if ALL_MODELS:
        make_models.append(menangerie.resnext50_32x4d_16)
        make_models.append(menangerie.gpt2_16)
        make_models.append(menangerie.gpt2)
        make_models.append(menangerie.mobilenet_16)
        make_models.append(menangerie.mobilenet)
        make_models.append(menangerie.resnet50_16)
        make_models.append(menangerie.resnet50)
    run_models = []
    if ALL_CONFIGS:
        run_models.append(just_tensorrt)
        run_models.append(just_tvm)
        run_models.append(tvm_with_libs)
    run_models.append(collage)
    for make_model in make_models:
        model = make_model()
        for run_model in run_models:
            run_model(model)


def run_mini():
    """Run Collage on a tiny GPT2 extract."""
    collage(menangerie.gpt2_16_for_cutlass_extract())


if __name__ == "__main__":
    # run_all()
    run_mini()
