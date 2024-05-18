#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of gpustat
@author Jongwook Choi
@url https://github.com/wookayin/gpustat

@ copied from gpu-stat 0.6.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import platform
import sys
import subprocess
from datetime import datetime
from ctypes import c_uint32, byref, c_int64

import psutil
from ..gpu import pynvml as N
from ..gpu import pyrsmi as R

NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024


class GPUStat(object):

    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError(
                'entry should be a dict, {} given'.format(type(entry))
            )
        self.entry = entry

    def keys(self):
        return self.entry.keys()

    def __getitem__(self, key):
        return self.entry[key]

    @property
    def index(self):
        """
        Returns the index of GPU (as in nvidia-smi).
        """
        return self.entry['index']

    @property
    def uuid(self):
        """
        Returns the uuid returned by nvidia-smi,
        e.g. GPU-12345678-abcd-abcd-uuid-123456abcdef
        """
        return self.entry['uuid']

    @property
    def mig_index(self):
        """
        Returns the index of the MIG partition (as in nvidia-smi).
        """
        return self.entry.get("mig_index")

    @property
    def mig_uuid(self):
        """
        Returns the uuid of the MIG partition returned by nvidia-smi when running in MIG mode,
        e.g. MIG-12345678-abcd-abcd-uuid-123456abcdef
        """
        return self.entry.get("mig_uuid")

    @property
    def name(self):
        """
        Returns the name of GPU card (e.g. Geforce Titan X)
        """
        return self.entry['name']

    @property
    def memory_total(self):
        """
        Returns the total memory (in MB) as an integer.
        """
        return int(self.entry['memory.total'])

    @property
    def memory_used(self):
        """
        Returns the occupied memory (in MB) as an integer.
        """
        return int(self.entry['memory.used'])

    @property
    def memory_free(self):
        """
        Returns the free (available) memory (in MB) as an integer.
        """
        v = self.memory_total - self.memory_used
        return max(v, 0)

    @property
    def memory_available(self):
        """
        Returns the available memory (in MB) as an integer.
        Alias of memory_free.
        """
        return self.memory_free

    @property
    def temperature(self):
        """
        Returns the temperature (in celcius) of GPU as an integer,
        or None if the information is not available.
        """
        v = self.entry['temperature.gpu']
        return int(v) if v is not None else None

    @property
    def fan_speed(self):
        """
        Returns the fan speed percentage (0-100) of maximum intended speed
        as an integer, or None if the information is not available.
        """
        v = self.entry['fan.speed']
        return int(v) if v is not None else None

    @property
    def utilization(self):
        """
        Returns the GPU utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.gpu']
        return int(v) if v is not None else None

    @property
    def power_draw(self):
        """
        Returns the GPU power usage in Watts,
        or None if the information is not available.
        """
        v = self.entry['power.draw']
        return int(v) if v is not None else None

    @property
    def power_limit(self):
        """
        Returns the (enforced) GPU power limit in Watts,
        or None if the information is not available.
        """
        v = self.entry['enforced.power.limit']
        return int(v) if v is not None else None

    @property
    def processes(self):
        """
        Get the list of running processes on the GPU.
        """
        return self.entry['processes']

    def jsonify(self):
        o = dict(self.entry)
        if self.entry['processes'] is not None:
            o['processes'] = [{k: v for (k, v) in p.items() if k != 'gpu_uuid'}
                              for p in self.entry['processes']]
        else:
            o['processes'] = '({})'.format(NOT_SUPPORTED)
        return o


class GPUStatCollection(object):
    global_processes = {}
    _initialized = False
    _device_count = None
    _gpu_device_info = {}
    _mig_device_info = {}

    def __init__(self, gpu_list, driver_version=None, driver_cuda_version=None):
        self.gpus = gpu_list

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()
        self.driver_version = driver_version
        self.driver_cuda_version = driver_cuda_version

    @staticmethod
    def clean_processes():
        for pid in list(GPUStatCollection.global_processes.keys()):
            if not psutil.pid_exists(pid):
                del GPUStatCollection.global_processes[pid]

    @staticmethod
    def _new_query_amd(shutdown=False, per_process_stats=False, get_driver_info=False):
        initialized = False
        if not GPUStatCollection._initialized:
            R.smi_initialize()
            GPUStatCollection._initialized = True
            initialized = True

        def get_gpu_info(index):
            def amd_query_processes():
                num_procs = c_uint32()
                ret = R.rocm_lib.rsmi_compute_process_info_get(None, byref(num_procs))
                if R.rsmi_ret_ok(ret):
                    buff_sz = num_procs.value + 10
                    proc_info = (R.rsmi_process_info_t * buff_sz)()
                    ret = R.rocm_lib.rsmi_compute_process_info_get(byref(proc_info), byref(num_procs))
                    proc_info_list = (
                        [proc_info[i] for i in range(num_procs.value)]
                        if R.rsmi_ret_ok(ret)
                        else []
                    )
                    result_proc_info_list = []
                    # query VRAM usage explicitly, as rsmi_compute_process_info_get
                    # doesn't actually return VRAM usage
                    for proc_info in proc_info_list:
                        vram_query_proc_info = R.rsmi_process_info_t()
                        ret = R.rocm_lib.rsmi_compute_process_info_by_pid_get(
                            int(proc_info.process_id), byref(vram_query_proc_info)
                        )
                        if R.rsmi_ret_ok(ret):
                            proc_info.vram_usage = vram_query_proc_info.vram_usage
                            result_proc_info_list.append(proc_info)
                    return result_proc_info_list
                return []

            def get_fan_speed():
                fan_level = c_int64()
                fan_max = c_int64()
                sensor_ind = c_uint32(0)

                ret = R.rocm_lib.rsmi_dev_fan_speed_get(index, sensor_ind, byref(fan_level))
                if not R.rsmi_ret_ok(ret, log_error=False):
                    return None

                ret = R.rocm_lib.rsmi_dev_fan_speed_max_get(index, sensor_ind, byref(fan_max))
                if not R.rsmi_ret_ok(ret, log_error=False):
                    return None

                if fan_level.value <= 0 or fan_max <= 0:
                    return None

                return float(fan_level.value) / float(fan_max.value)

            def get_process_info(comp_process):
                process = {}
                pid = comp_process.process_id
                # skip global_processes caching because PID querying seems to be inconsistent atm
                # if pid not in GPUStatCollection.global_processes:
                #     GPUStatCollection.global_processes[pid] = psutil.Process(pid=pid)
                process["pid"] = pid
                try:
                    process["gpu_memory_usage"] = comp_process.vram_usage // MB
                except Exception:
                    pass
                return process

            if not GPUStatCollection._gpu_device_info.get(index):
                uuid = R.smi_get_device_id(index)
                name = R.smi_get_device_name(index)
                GPUStatCollection._gpu_device_info[index] = (name, uuid)

            name, uuid = GPUStatCollection._gpu_device_info[index]

            temperature = None  # TODO: fetch temperature. It should be possible
            fan_speed = get_fan_speed()

            try:
                memory_total = R.smi_get_device_memory_total(index)
            except Exception:
                memory_total = None

            try:
                memory_used = R.smi_get_device_memory_used(index)
            except Exception:
                memory_used = None

            try:
                utilization = R.smi_get_device_utilization(index)
            except Exception:
                utilization = None

            try:
                power = R.smi_get_device_average_power(index)
            except Exception:
                power = None

            power_limit = None  # TODO: find a way to fetch this

            processes = []
            if per_process_stats:
                try:
                    comp_processes = amd_query_processes()
                except Exception:
                    comp_processes = []
                for comp_process in comp_processes:
                    try:
                        process = get_process_info(comp_process)
                    except psutil.NoSuchProcess:
                        # skip process caching for now
                        pass
                    else:
                        processes.append(process)

            gpu_info = {
                "index": index,
                "uuid": uuid,
                "name": name,
                "temperature.gpu": temperature if temperature is not None else 0,
                "fan.speed": fan_speed if fan_speed is not None else 0,
                "utilization.gpu": utilization if utilization is not None else 100,
                "power.draw": power if power is not None else 0,
                "enforced.power.limit": power_limit if power_limit is not None else 0,
                # Convert bytes into MBytes
                "memory.used": memory_used // MB if memory_used is not None else 0,
                "memory.total": memory_total // MB if memory_total is not None else 100,
                "processes": None if (processes and all(p is None for p in processes)) else processes,
            }
            if per_process_stats:
                GPUStatCollection.clean_processes()
            return gpu_info

        gpu_list = []
        if GPUStatCollection._device_count is None:
            GPUStatCollection._device_count = R.smi_get_device_count()

        for index in range(GPUStatCollection._device_count):
            gpu_info = get_gpu_info(index)
            gpu_stat = GPUStat(gpu_info)
            gpu_list.append(gpu_stat)

        if shutdown and initialized:
            R.smi_shutdown()
            GPUStatCollection._initialized = False

        # noinspection PyProtectedMember
        driver_version = GPUStatCollection._get_amd_driver_version() if get_driver_info else None

        return GPUStatCollection(gpu_list, driver_version=driver_version, driver_cuda_version=None)

    @staticmethod
    def _get_amd_driver_version():
        # make sure the program doesn't crash with something like a SEGFAULT when querying the driver version
        try:
            process = subprocess.Popen(["rocm-smi", "--showdriverversion", "--json"], stdout=subprocess.PIPE)
            out, _ = process.communicate()
            return json.loads(out)["system"]["Driver version"]
        except Exception:
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "from clearml.utilities.gpu.pyrsmi import smi_get_kernel_version, smi_initialize; "
                        + "smi_initialize(); "
                        + "print(smi_get_kernel_version())",
                    ]
                )
                out, _ = process.communicate()
                return out.strip()
            except Exception:
                return None

    @staticmethod
    def _running_in_amd_env():
        # noinspection PyProtectedMember
        return bool(R._find_lib_rocm())

    @staticmethod
    def _new_query_nvidia(shutdown=False, per_process_stats=False, get_driver_info=False):
        """Query the information of all the GPUs on local machine"""
        initialized = False
        if not GPUStatCollection._initialized:
            N.nvmlInit()
            GPUStatCollection._initialized = True
            initialized = True

        def _decode(b):
            if isinstance(b, bytes):
                return b.decode()  # for python3, to unicode
            return b

        def get_gpu_info(index, handle, is_mig=False):
            """Get one GPU information specified by nvml handle"""

            def get_process_info(nv_process):
                """Get the process information of specific pid"""
                process = {}
                if nv_process.pid not in GPUStatCollection.global_processes:
                    GPUStatCollection.global_processes[nv_process.pid] = \
                        psutil.Process(pid=nv_process.pid)
                process['pid'] = nv_process.pid
                # noinspection PyBroadException
                try:
                    # ps_process = GPUStatCollection.global_processes[nv_process.pid]
                    # we do not actually use these, so no point in collecting them
                    # process['username'] = ps_process.username()
                    # # cmdline returns full path;
                    # # as in `ps -o comm`, get short cmdnames.
                    # _cmdline = ps_process.cmdline()
                    # if not _cmdline:
                    #     # sometimes, zombie or unknown (e.g. [kworker/8:2H])
                    #     process['command'] = '?'
                    #     process['full_command'] = ['?']
                    # else:
                    #     process['command'] = os.path.basename(_cmdline[0])
                    #     process['full_command'] = _cmdline
                    # process['cpu_percent'] = ps_process.cpu_percent()
                    # process['cpu_memory_usage'] = \
                    #     round((ps_process.memory_percent() / 100.0) *
                    #           psutil.virtual_memory().total)
                    # Bytes to MBytes
                    process['gpu_memory_usage'] = nv_process.usedGpuMemory // MB
                except Exception:
                    # insufficient permissions
                    pass
                return process

            device_info = GPUStatCollection._mig_device_info if is_mig else GPUStatCollection._gpu_device_info
            if not device_info.get(index):
                name = _decode(N.nvmlDeviceGetName(handle))
                uuid = _decode(N.nvmlDeviceGetUUID(handle))
                device_info[index] = (name, uuid)

            name, uuid = device_info[index]

            try:
                temperature = N.nvmlDeviceGetTemperature(
                    handle, N.NVML_TEMPERATURE_GPU
                )
            except N.NVMLError:
                temperature = None  # Not supported

            try:
                fan_speed = N.nvmlDeviceGetFanSpeed(handle)
            except N.NVMLError:
                fan_speed = None  # Not supported

            try:
                memory = N.nvmlDeviceGetMemoryInfo(handle)  # in Bytes
            except N.NVMLError:
                memory = None  # Not supported

            try:
                utilization = N.nvmlDeviceGetUtilizationRates(handle)
            except N.NVMLError:
                utilization = None  # Not supported

            try:
                power = N.nvmlDeviceGetPowerUsage(handle)
            except N.NVMLError:
                power = None

            try:
                power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
            except N.NVMLError:
                power_limit = None

            try:
                nv_comp_processes = \
                    N.nvmlDeviceGetComputeRunningProcesses(handle)
            except N.NVMLError:
                nv_comp_processes = None  # Not supported
            try:
                nv_graphics_processes = \
                    N.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except N.NVMLError:
                nv_graphics_processes = None  # Not supported

            if not per_process_stats or (nv_comp_processes is None and nv_graphics_processes is None):
                processes = None
            else:
                processes = []
                nv_comp_processes = nv_comp_processes or []
                nv_graphics_processes = nv_graphics_processes or []
                for nv_process in nv_comp_processes + nv_graphics_processes:
                    try:
                        process = get_process_info(nv_process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        process = None
                    processes.append(process)

                # we do not actually use these, so no point in collecting them
                # # TODO: Do not block if full process info is not requested
                # time.sleep(0.1)
                # for process in processes:
                #     pid = process['pid']
                #     cache_process = GPUStatCollection.global_processes[pid]
                #     process['cpu_percent'] = cache_process.cpu_percent()

            index = N.nvmlDeviceGetIndex(handle)
            gpu_info = {
                'index': index,
                'uuid': uuid,
                'name': name,
                'temperature.gpu': temperature,
                'fan.speed': fan_speed,
                'utilization.gpu': utilization.gpu if utilization else None,
                'power.draw': power // 1000 if power is not None else None,
                'enforced.power.limit': power_limit // 1000
                if power_limit is not None else None,
                # Convert bytes into MBytes
                'memory.used': memory.used // MB if memory else None,
                'memory.total': memory.total // MB if memory else None,
                'processes': None if (processes and all(p is None for p in processes)) else processes
            }
            if per_process_stats:
                GPUStatCollection.clean_processes()
            return gpu_info

        # 1. get the list of gpu and status
        gpu_list = []
        if GPUStatCollection._device_count is None:
            GPUStatCollection._device_count = N.nvmlDeviceGetCount()

        for index in range(GPUStatCollection._device_count):
            handle = N.nvmlDeviceGetHandleByIndex(index)
            gpu_info = get_gpu_info(index, handle)
            mig_cnt = 0
            # noinspection PyBroadException
            try:
                mig_cnt = N.nvmlDeviceGetMaxMigDeviceCount(handle)
            except Exception:
                pass

            if mig_cnt <= 0:
                gpu_list.append(GPUStat(gpu_info))
                continue

            got_mig_info = False
            for mig_index in range(mig_cnt):
                # noinspection PyBroadException
                try:
                    mig_handle = N.nvmlDeviceGetMigDeviceHandleByIndex(handle, mig_index)
                    mig_info = get_gpu_info(mig_index, mig_handle, is_mig=True)
                    mig_info["mig_name"] = mig_info["name"]
                    mig_info["name"] = gpu_info["name"]
                    mig_info["mig_index"] = mig_info["index"]
                    mig_info["mig_uuid"] = mig_info["uuid"]
                    mig_info["index"] = gpu_info["index"]
                    mig_info["uuid"] = gpu_info["uuid"]
                    mig_info["temperature.gpu"] = gpu_info["temperature.gpu"]
                    mig_info["fan.speed"] = gpu_info["fan.speed"]
                    gpu_list.append(GPUStat(mig_info))
                    got_mig_info = True
                except Exception:
                    pass
            if not got_mig_info:
                gpu_list.append(GPUStat(gpu_info))

        # 2. additional info (driver version, etc).
        if get_driver_info:
            try:
                driver_version = _decode(N.nvmlSystemGetDriverVersion())
            except N.NVMLError:
                driver_version = None  # N/A

            # noinspection PyBroadException
            try:
                cuda_driver_version = str(N.nvmlSystemGetCudaDriverVersion())
            except BaseException:
                # noinspection PyBroadException
                try:
                    cuda_driver_version = str(N.nvmlSystemGetCudaDriverVersion_v2())
                except BaseException:
                    cuda_driver_version = None
            if cuda_driver_version:
                try:
                    cuda_driver_version = '{}.{}'.format(
                        int(cuda_driver_version)//1000, (int(cuda_driver_version) % 1000)//10)
                except (ValueError, TypeError):
                    pass
        else:
            driver_version = None
            cuda_driver_version = None

        # no need to shutdown:
        if shutdown and initialized:
            N.nvmlShutdown()
            GPUStatCollection._initialized = False

        return GPUStatCollection(gpu_list, driver_version=driver_version, driver_cuda_version=cuda_driver_version)

    @staticmethod
    def new_query(shutdown=False, per_process_stats=False, get_driver_info=False):
        # noinspection PyProtectedMember
        if GPUStatCollection._running_in_amd_env():
            # noinspection PyProtectedMember
            return GPUStatCollection._new_query_amd(
                shutdown=shutdown, per_process_stats=per_process_stats, get_driver_info=get_driver_info
            )
        else:
            # noinspection PyProtectedMember
            return GPUStatCollection._new_query_nvidia(
                shutdown=shutdown, per_process_stats=per_process_stats, get_driver_info=get_driver_info
            )

    def __len__(self):
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, index):
        return self.gpus[index]

    def __repr__(self):
        s = 'GPUStatCollection(host=%s, [\n' % self.hostname
        s += '\n'.join('  ' + str(g) for g in self.gpus)
        s += '\n])'
        return s

    # --- Printing Functions ---
    def jsonify(self):
        return {
            'hostname': self.hostname,
            'query_time': self.query_time,
            "gpus": [g.jsonify() for g in self]
        }

    def print_json(self, fp=sys.stdout):
        def date_handler(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                raise TypeError(type(obj))

        o = self.jsonify()
        json.dump(o, fp, indent=4, separators=(',', ': '),
                  default=date_handler)
        fp.write('\n')
        fp.flush()


def new_query(shutdown=False, per_process_stats=False, get_driver_info=False):
    '''
    Obtain a new GPUStatCollection instance by querying nvidia-smi
    to get the list of GPUs and running process information.
    '''
    return GPUStatCollection.new_query(shutdown=shutdown, per_process_stats=per_process_stats,
                                       get_driver_info=get_driver_info)
