import jax

grad_tanh = jax.grad(jax.numpy.tanh)

# for i in range(1, 100):
#     print(1/i, ": ", grad_tanh(1/i))

print("\n\n\n***************************************\n", grad_tanh(0.2))