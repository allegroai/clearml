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
from datetime import datetime

import psutil
from ..gpu import pynvml as N

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
    def new_query(shutdown=False, per_process_stats=False, get_driver_info=False):
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

        def get_gpu_info(index, handle):
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

            if not GPUStatCollection._gpu_device_info.get(index):
                name = _decode(N.nvmlDeviceGetName(handle))
                uuid = _decode(N.nvmlDeviceGetUUID(handle))
                GPUStatCollection._gpu_device_info[index] = (name, uuid)

            name, uuid = GPUStatCollection._gpu_device_info[index]

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
                        processes.append(process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        pass

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
                'processes': processes,
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
            gpu_stat = GPUStat(gpu_info)
            gpu_list.append(gpu_stat)

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
