#include "renorm/kernel_context.h"

namespace renorm {

KernelContext::KernelContext()
    : kernel_calls_(0)
{
}

KernelContext::~KernelContext() = default;

WorkspaceAllocator& KernelContext::workspace()
{
    return workspace_;
}

std::uint64_t KernelContext::kernel_calls() const
{
    return kernel_calls_;
}

void KernelContext::increment_kernel_calls()
{
    ++kernel_calls_;
}

void KernelContext::reset_statistics()
{
    kernel_calls_ = 0;
}

} // namespace renorm