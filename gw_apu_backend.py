import sys
import torch
print("torch version=", torch.version.__version__)


print(torch.backends)
print(dir(torch.backends))


# Check that APU is available
if not torch.backends.apu.is_available():
    if not torch.backends.apu.is_built():
        print("APU not available because the current PyTorch install was not "
              "built with APU enabled.")

else:
    apu_device = torch.device("privateuseone")

    # Create a Tensor directly on the apu device
    x = torch.ones(5, device=apu_device)
    # Or
    #x = torch.ones(5, device="apu")

    # Any operation happens on the GPU
    y = x * 2
    print(y)

