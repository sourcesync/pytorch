#include <ATen/Context.h>
#include <torch/apu.h>

namespace torch {
namespace apu {

bool is_available() {
  //GW return at::detail::getAPUHooks().hasAPU();
  return true;
}

/// Sets the seed for the APU's default generator.
void manual_seed(uint64_t seed) {
  if (is_available()) {
    /* GW
    auto gen = at::detail::getAPUHooks().getDefaultAPUGenerator();
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
    */
  }
}

void synchronize() {
  //GW at::detail::getAPUHooks().deviceSynchronize();
}

void commit() {
  //GW at::detail::getAPUHooks().commitStream();
}

/* GW
MTLCommandBuffer_t get_command_buffer() {
  //GW return at::detail::getAPUHooks().getCommandBuffer();
}

DispatchQueue_t get_dispatch_queue() {
  //GW return at::detail::getAPUHooks().getDispatchQueue();
}
*/

} // namespace apu
} // namespace torch
