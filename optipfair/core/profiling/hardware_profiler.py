import torch.cuda
import platform
from core.exceptions.platform_not_supported import PlatformNotSupportedException
import psutil
from core.profiling.types.hardware import (
    CPUInfo,
    GPUInfo,
    SystemGPUInfo,
    GPUProperties,
    GPUMemoryInfo,
    RAMInfo,
    HardwareProfile,
)
from datetime import datetime


class HardwareProfiler:
    def retrive_hardware_information(self) -> HardwareProfile:
        cpu_info = self.retrieve_cpu_information()
        gpu_info = self.retrieve_gpu_information()
        ram_info = self.retrieve_ram_information()

        return HardwareProfile(cpu=cpu_info, gpu=gpu_info, ram=ram_info)

    def retrieve_cpu_information(self) -> CPUInfo:
        current_platform = platform.system()
        if current_platform == "Linux":
            return self.__get_linux_cpu_info()

        else:
            raise PlatformNotSupportedException(
                f"{current_platform} is not supported yet"
            )

    def __get_linux_cpu_info(self) -> CPUInfo:
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.strip().split(": ")[1]
        except (FileNotFoundError, IndexError) as e:
            raise e

        return CPUInfo(
            architecture=platform.processor(),
            name=cpu_name,
            max_freq=psutil.cpu_freq().max,
            platform=platform.platform(),
            physical_cores=psutil.cpu_count(logical=False),
            total_cores=psutil.cpu_count(logical=True),
        )

    def retrieve_gpu_information(self) -> SystemGPUInfo:
        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            return SystemGPUInfo(
                cuda_available=False,
                cuda_version=None,
                cudnn_version=None,
                device_count=0,
                current_device=None,
                gpus=[],
                timestamp=datetime.now(),
            )

        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device() if device_count > 0 else None

        cuda_version = torch.version.cuda
        cudnn_version = (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        )

        gpus = list()
        for device in range(device_count):
            try:
                gpu_info = self._get_gpu_info(device)
                gpus.append(gpu_info)
            except Exception as e:
                print(f"Warning: Could not get info for GPU {device}: {e}")

        return SystemGPUInfo(
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            device_count=device_count,
            current_device=current_device,
            gpus=gpus,
            timestamp=datetime.now(),
        )

    def _get_gpu_info(self, device_id: int) -> GPUInfo:
        """Get complete information for a specific GPU device"""
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        if device_id >= torch.cuda.device_count():
            raise ValueError(f"GPU device {device_id} is not available")

        device_name = torch.cuda.get_device_name(device_id)
        memory_info = self._get_gpu_memory_info(device_id)
        properties = self._get_gpu_properties(device_id)

        return GPUInfo(
            device_id=device_id,
            device_name=device_name,
            is_available=True,
            memory_info=memory_info,
            properties=properties,
        )

    def _bytes_to_gb(self, bytes_value: int) -> float:
        return round(bytes_value / (1024**3), 2)

    def _get_gpu_memory_info(self, device_id: int) -> GPUMemoryInfo:
        """Get memory information for a specific GPU device"""
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            raise ValueError(f"GPU device {device_id} is not available")

        with torch.cuda.device(device_id):
            total: int = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            cached = torch.cuda.memory_reserved(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            free = total - allocated

            return GPUMemoryInfo(
                total=total,
                allocated=allocated,
                cached=cached,
                reserved=reserved,
                free=free,
                total_gb=self._bytes_to_gb(total),
                allocated_gb=self._bytes_to_gb(allocated),
                cached_gb=self._bytes_to_gb(cached),
                reserved_gb=self._bytes_to_gb(reserved),
                free_gb=self._bytes_to_gb(free),
            )

    def _get_gpu_properties(self, device_id: int) -> GPUProperties:
        """Get properties for a specific GPU device"""
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            raise ValueError(f"GPU device {device_id} is not available")

        props = torch.cuda.get_device_properties(device_id)

        return GPUProperties(
            name=props.name,
            major=props.major,
            minor=props.minor,
            total_memory=props.total_memory,
            multi_processor_count=props.multi_processor_count,
            max_threads_per_multi_processor=props.max_threads_per_multi_processor,
            max_threads_per_block=props.max_threads_per_block,
            max_block_dim=list(props.max_block_dim),
            max_grid_dim=list(props.max_grid_dim),
            warp_size=props.warp_size,
        )

    def retrieve_ram_information(self) -> RAMInfo:
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()

        return RAMInfo(
            total_memory_gb=self._bytes_to_gb(virtual_memory.total),
            available_memory_gb=self._bytes_to_gb(virtual_memory.available),
            free_memory_gb=self._bytes_to_gb(virtual_memory.free),
            swap_total_gb=self._bytes_to_gb(swap_memory.total),
            swap_free_gb=self._bytes_to_gb(swap_memory.free),
        )
