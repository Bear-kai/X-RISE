# 🖥️ Installation guide

## Conda Environment for X-RISE

- If you do not need to evaluate on RoboTwin 2.0 benchmark, please follow the [guide](https://github.com/rise-policy/RISE/blob/main/assets/docs/INSTALL.md) to install the `rise` conda environment, since X-RISE is based on RISE.

- If you have installed the `robotwin` conda environment, basicly you only need to install the MinkowskiEngine as in the [guide](https://github.com/rise-policy/RISE/blob/main/assets/docs/INSTALL.md) and the following packages:
    ```bash
    pip install easydict==1.13 einops==0.4.1 diffusers==0.11.1
    ```

- During our last test, the commit ID of robotwin repository was `ea25643210c3036e57eb44b4cd4b59b8dc1e98d7`.

- We have tested on `Ubuntu 22.04` + `CUDA 12.1` + `python 3.10` + `torch 2.4.1`. It is the recommended environment but others may work.

## GPU issues

Please note that the `cuRobo` package (required by evaluation on RoboTwin 2.0) requires Ubuntu >= 20.04 and an NVIDIA GPU newer than the VOLTA architecture. Therefore, it is recommended to use a compatible GPU, such as an RTX 4090. In case you have to work with an older GPU like a Titan XP, we provide the following workaround.

- modify the `envs/curobo/src/curobo/curobolib/cpp/lbfgs_step_kernel.cu` file:

    <details>
    <summary>detail</summary>

    **before (lines 893-907):**
    ```cpp
    // try to increase shared memory:
    // Note that this feature is only available from volta+ (cuda 7.0+)
    #if (VOLTA_PLUS)
    {
    if (use_shared_buffers && max_shared_increase > max_shared_base && max_shared_increase <= max_shared_allowed)
    { 
        max_shared = max_shared_increase;
        cudaError_t result;
        result = cudaFuncSetAttribute(selected_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_increase);
        if (result != cudaSuccess)
        {
        max_shared = max_shared_base;
        }
    }
    }
    #endif
    ```

    **after**:
    ```cpp
    // try to increase shared memory:
    // Note that this feature is only available from volta+ (cuda 7.0+)
    if (use_shared_buffers && max_shared_increase > max_shared_base && max_shared_increase <= max_shared_allowed)
    {
        cudaError_t result;
        result = cudaFuncSetAttribute(selected_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_increase);
        if (result == cudaSuccess)
        {
            max_shared = max_shared_increase;
        }
        else
        {
            cudaGetLastError(); // Clear error state on pre-Volta GPUs that don't support this attribute
        }
    }
    ```

    </details>

- modify the `envs/_base_task.py` file:
    ```python
    # sapien.render.set_ray_tracing_denoiser("oidn")   # before
    sapien.render.set_ray_tracing_denoiser("optix")    # after
    ```