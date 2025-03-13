import logging

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# import the reference implementations
from pytorch_reference_backwards import (
    _compute_grad_w_pytorch,
    _compute_grad_x_pytorch,
    _pytorch_fallback_backward,
    _pytorch_reference_backward,
)

# Import the grouped GEMM modules
from tgrouped_gemm_backwards import grouped_gemm_backward
from tgrouped_gemm_forward import grouped_gemm_forward as grouped_gemm


def test_backward_pass():
    """
    A simple test for the grouped GEMM backward pass with detailed error handling.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test parameters
        G = 20  # Number of groups
        M = 1024  # Input dimension
        N = 512  # Output dimension per group
        K = 256  # Hidden dimension

        # Create input and weight tensors
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
        w = torch.randn(
            N * G, K, dtype=torch.bfloat16, device=device, requires_grad=True
        )

        # Create group sizes
        m_sizes = torch.zeros(G, device=device, dtype=torch.int32)
        base_size = M // G
        remainder = M % G

        for i in range(G):
            m_sizes[i] = base_size + (1 if i < remainder else 0)

        # Log the setup
        print(f"Test setup - G: {G}, M: {M}, N: {N}, K: {K}")
        print(f"Input x shape: {x.shape}")
        logging.info(f"Weight w shape: {w.shape}")
        logging.info(f"Group sizes: {m_sizes}")

        # Step 1: Run forward pass
        logging.info("Running forward pass")
        result = grouped_gemm(x, w, m_sizes)
        logging.info(f"Forward result shape: {result.shape}")

        # Create a gradient for backpropagation
        grad_output = torch.randn_like(result)
        logging.info(f"Created gradient with shape: {grad_output.shape}")

        # Step 2: Run backward pass directly
        logging.info("Running backward pass directly")
        grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)

        # Verify gradient shapes
        logging.info(
            f"Gradient shapes - grad_x: {grad_x.shape}, grad_w: {grad_w.shape}"
        )

        # Step 3: Verify gradient computation using PyTorch's autograd
        # First create autograd-enabled tensors
        x_autograd = x.detach().clone().requires_grad_(True)
        w_autograd = w.detach().clone().requires_grad_(True)

        # Create a PyTorch reference implementation to compare against
        logging.info("Running PyTorch reference implementation")

        # Compute reference result
        reference_result = torch.zeros_like(result)
        m_start = 0
        for g in range(G):
            m_size = m_sizes[g].item()
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            if m_size > 0:
                reference_result[m_start:m_end, n_start:n_end] = (
                    x_autograd[m_start:m_end, :] @ w_autograd[n_start:n_end, :].T
                )

            m_start = m_end

        # Backpropagate using PyTorch
        reference_result.backward(grad_output)

        # Compare gradients
        logging.info("Comparing gradients with PyTorch reference")
        grad_x_error = (grad_x - x_autograd.grad).abs().max().item()
        grad_w_error = (grad_w - w_autograd.grad).abs().max().item()

        logging.info(
            f"Maximum gradient error - grad_x: {grad_x_error}, grad_w: {grad_w_error}"
        )

        # Check if gradients are close using allclose
        rtol = 1e-2  # Relative tolerance for bfloat16
        atol = 1e-2  # Absolute tolerance for bfloat16

        grad_x_close = torch.allclose(grad_x, x_autograd.grad, rtol=rtol, atol=atol)
        if not grad_x_close:
            logging.warning("FAILED: Gradient mismatch detected in grad_x")
        else:
            logging.info(
                "✓ SUCCESS! grad_X matches the PyTorch reference (allclose check passed)"
            )

        grad_w_close = torch.allclose(grad_w, w_autograd.grad, rtol=rtol, atol=atol)
        if not grad_w_close:
            logging.warning("FAILED: Gradient mismatch detected in grad_w")
        else:
            logging.info(
                "✓ SUCCESS! grad_W matches the PyTorch reference (allclose check passed)"
            )

        logging.info(
            f"Gradients allclose check - grad_x: {grad_x_close}, grad_w: {grad_w_close}"
        )

        if grad_x_close and grad_w_close:
            logging.info(
                "✓ SUCCESS: Gradients match the PyTorch reference (allclose check passed)"
            )
        else:
            logging.error("✗ FAILURE: Gradient mismatch detected in allclose check")

        # Additional diagnostics (for failed cases or in general)
        if True:  # not grad_x_close:
            # Find where the largest differences are
            diff_x = (grad_x - x_autograd.grad).abs()
            max_idx_x = diff_x.argmax().item()
            flat_idx_x = max_idx_x
            idx_x = np.unravel_index(flat_idx_x, grad_x.shape)
            logging.error(
                f"Largest grad_x difference at {idx_x}: "
                f"{grad_x[idx_x].item()} vs {x_autograd.grad[idx_x].item()}"
            )
            # Count zeros
            zeros_grad_x = (grad_x == 0).sum().item()
            zeros_autograd_x = (x_autograd.grad == 0).sum().item()
            logging.error(
                f"Zeros in grad_x: {zeros_grad_x}/{grad_x.numel()} ({zeros_grad_x/grad_x.numel()*100:.2f}%)"
            )
            logging.error(
                f"Zeros in x_autograd.grad: {zeros_autograd_x}/{x_autograd.grad.numel()} ({zeros_autograd_x/x_autograd.grad.numel()*100:.2f}%)"
            )

        if True:  # not grad_w_close:
            # Find where the largest differences are
            diff_w = (grad_w - w_autograd.grad).abs()
            max_idx_w = diff_w.argmax().item()
            flat_idx_w = max_idx_w
            idx_w = np.unravel_index(flat_idx_w, grad_w.shape)
            logging.error(
                f"Largest grad_w difference at {idx_w}: "
                f"{grad_w[idx_w].item()} vs {w_autograd.grad[idx_w].item()}"
            )
            # Count zeros
            zeros_grad_w = (grad_w == 0).sum().item()
            zeros_autograd_w = (w_autograd.grad == 0).sum().item()
            logging.error(
                f"Zeros in grad_w: {zeros_grad_w}/{grad_w.numel()} ({zeros_grad_w/grad_w.numel()*100:.2f}%)"
            )
            logging.error(
                f"Zeros in w_autograd.grad: {zeros_autograd_w}/{w_autograd.grad.numel()} ({zeros_autograd_w/w_autograd.grad.numel()*100:.2f}%)"
            )

        return grad_x_close and grad_w_close

    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("Running test_backward_pass")
    logging.debug("Running test_backward_pass")
    # Add numpy import for unravel_index
    import numpy as np

    success = test_backward_pass()
    logging.info(f"Test {'succeeded' if success else 'failed'}")
