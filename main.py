def generate_rsa_keypair(key_size=128):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    """
    Generate RSA keypair (p, q, e, d) with a specified key size.
    :param key_size: Key size in bits. Should be a multiple of 128.
    :return: Tuple (p, q, n, e, d) where:
             - p, q: Prime numbers
             - n: Modulus (p * q)
             - e: Public exponent
             - d: Private exponent
    """
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,  # Commonly used public exponent
        key_size=key_size,      # Key size (e.g., 128 bits for testing)
        backend=default_backend()
    )
    
    # Extract components
    private_numbers = private_key.private_numbers()
    public_numbers = private_key.public_key().public_numbers()
    
    p = private_numbers.p
    q = private_numbers.q
    n = public_numbers.n
    e = public_numbers.e
    d = private_numbers.d

    return p, q, n, e, d


import pyopencl as cl
import numpy as np

class ParallelRSA512:
    CHUNK_SIZE = 8  # Number of 64-bit chunks for 512-bit numbers

    def __init__(self, platform_index=0, device_index=0):
        # OpenCL context setup
        self.platform = cl.get_platforms()[platform_index]
        self.device = self.platform.get_devices()[device_index]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        # Load and build the OpenCL kernel
        with open("rsa_kernels_512.cl", "r") as kernel_file:
            self.program = cl.Program(self.context, kernel_file.read()).build()

    def convert_to_chunks(self, numbers):
        """
        Convert a list of large integers into arrays of 64-bit chunks.
        """
        chunked = np.zeros((len(numbers), self.CHUNK_SIZE), dtype=np.uint64)
        for i, num in enumerate(numbers):
            for j in range(self.CHUNK_SIZE):
                chunked[i, j] = num & 0xFFFFFFFFFFFFFFFF
                num >>= 64
        return chunked

    def convert_from_chunks(self, chunked):
        """
        Convert arrays of 64-bit chunks back into large integers.
        """
        numbers = []
        for chunks in chunked:
            num = 0
            for i in reversed(range(self.CHUNK_SIZE)):
                num = (num << 64) | chunks[i]
            numbers.append(num)
        return np.array(numbers, dtype=np.uint64)

    def rsa_operation(self, numbers, exponents, mod):
        """
        Perform modular exponentiation: result = numbers^exponents % mod
        :param numbers: List of large integers (bases)
        :param exponents: List of large integers (exponents)
        :param mod: Single large integer (modulus)
        :return: List of large integers (results)
        """
        chunked_numbers = self.convert_to_chunks(numbers)
        chunked_exponents = self.convert_to_chunks(exponents)
        chunked_mod = self.convert_to_chunks([mod])[0]

        print(chunked_numbers)
        print(chunked_exponents)
        print(chunked_mod)

        # Prepare OpenCL buffers
        numbers_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=chunked_numbers)
        exponents_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=chunked_exponents)
        result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=chunked_numbers.nbytes)
        mod_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=chunked_mod)

        # Set kernel arguments and run
        kernel = self.program.mod_exp_512
        kernel.set_args(numbers_buf, exponents_buf, mod_buf, result_buf, np.uint32(len(numbers)))

        print((len(numbers),))
        cl.enqueue_nd_range_kernel(self.queue, kernel, (len(numbers),), None)
        results = np.empty_like(chunked_numbers)
        cl.enqueue_copy(self.queue, results, result_buf)

        return self.convert_from_chunks(results)

    def encrypt(self, messages, e, n):
        return self.rsa_operation(messages, [e] * len(messages), n)

    def decrypt(self, ciphertexts, d, n):
        return self.rsa_operation(ciphertexts, [d] * len(ciphertexts), n)

class ParallelRSA128:
    def __init__(self, platform_index=0, device_index=0):
        self.platform = cl.get_platforms()[platform_index]
        self.device = self.platform.get_devices()[device_index]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        with open("rsa_kernels_128.cl", "r") as kernel_file:
            self.program = cl.Program(self.context, kernel_file.read()).build()

    def pack_128(self, numbers):
        """Pack 128-bit integers into (low, high) 64-bit components."""
        low = np.array([n & 0xFFFFFFFFFFFFFFFF for n in numbers], dtype=np.uint64)
        high = np.array([n >> 64 for n in numbers], dtype=np.uint64)
        return low, high

    def unpack_128(self, low, high):
        """Unpack 64-bit components back into 128-bit integers."""
        return [(h << 64) | l for h, l in zip(high, low)]

    def encrypt(self, messages, e, n):
        """
        Encrypt messages: C = M^e mod n for 128-bit RSA.
        """
        low_m, high_m = self.pack_128(messages)
        low_n, high_n = self.pack_128([n])
        results_low = np.zeros_like(low_m, dtype=np.uint64)
        results_high = np.zeros_like(high_m, dtype=np.uint64)

        # Create buffers
        m_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=low_m)
        m_high_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=high_m)
        n_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=low_n)
        n_high_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=high_n)
        result_low_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=results_low.nbytes)
        result_high_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=results_high.nbytes)

        # Set kernel arguments
        kernel = self.program.mod_exp_128
        kernel.set_args(m_buf, m_high_buf, np.uint64(e), n_buf, n_high_buf, result_low_buf, result_high_buf, np.uint32(len(low_m)))

        # Execute kernel
        cl.enqueue_nd_range_kernel(self.queue, kernel, (len(low_m),), None)
        cl.enqueue_copy(self.queue, results_low, result_low_buf)
        cl.enqueue_copy(self.queue, results_high, result_high_buf)

        return self.unpack_128(results_low, results_high)

    def decrypt(self, ciphertexts, d, n):
        return self.encrypt(ciphertexts, d, n)

class ParallelRSA2048:
    def __init__(self, platform_index=0, device_index=0):
        self.platform = cl.get_platforms()[platform_index]
        self.device = self.platform.get_devices()[device_index]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        # Assume kernel file contains the provided kernel code.
        with open("rsa_kernels_2048.cl", "r") as kernel_file:
            self.program = cl.Program(self.context, kernel_file.read()).build()

    def _prepare_data(self, data, chunk_size=32):
        """Convert large integers into arrays of smaller chunks."""
        chunks = []
        for num in data:
            chunk = []
            for _ in range(chunk_size):
                chunk.append(np.uint64(num & 0xFFFFFFFFFFFFFFFF))
                num >>= 64
            chunks.append(chunk)
        return np.array(chunks, dtype=np.uint64)

    def _combine_chunks(self, data):
        """Combine chunk arrays back into integers."""
        result = []
        for chunks in data:
            value = np.uint64(0)
            for i in reversed(range(len(chunks))):
                value = (value << np.uint64(64)) | chunks[i]
            result.append(int(value))  # Convert to Python int
        return result

    def encrypt(self, messages, exponent, modulus):
        """Encrypt messages using RSA."""
        # Prepare inputs
        messages_chunks = self._prepare_data(messages)
        exponent_chunks = self._prepare_data([exponent] * len(messages))
        modulus_chunks = self._prepare_data([modulus])

        # Create buffers
        mf = cl.mem_flags
        num_messages = len(messages)
        base_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=messages_chunks)
        exponent_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=exponent_chunks)
        modulus_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=modulus_chunks[0])
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, messages_chunks.nbytes)

        # Execute kernel
        global_size = (num_messages,)
        local_size = None
        self.program.rsa_mod_exp(
            self.queue, global_size, local_size,
            base_buf, exponent_buf, modulus_buf, result_buf,
            np.uint32(num_messages)
        )

        # Retrieve results
        result_chunks = np.zeros_like(messages_chunks)
        cl.enqueue_copy(self.queue, result_chunks, result_buf)

        # Combine chunks back to integers
        return self._combine_chunks(result_chunks)

    def decrypt(self, ciphertexts, private_exponent, modulus):
        """Decrypt ciphertexts using RSA."""
        # RSA decryption is similar to encryption with the private exponent
        return self.encrypt(ciphertexts, private_exponent, modulus)


if __name__ == "__main__":
    rsa2048 = ParallelRSA2048()
    
    p, q, n, e, d = generate_rsa_keypair(2048)

    # Print generated values
    print(f"p (prime): {p}")
    print(f"q (prime): {q}")
    print(f"n (modulus): {n}")
    print(f"e (public exponent): {e}")
    print(f"d (private exponent): {d}")

    # Plaintext messages (2048-bit integers)
    messages = [
        0x123456789ABCDEF,
    ]
    print("messages:", [hex(c) for c in messages])

    # Encrypt and decrypt
    ciphertexts = rsa2048.encrypt(messages, e, n)
    print("Ciphertexts:", [hex(c) for c in ciphertexts])

    decrypted_messages = rsa2048.decrypt(ciphertexts, d, n)
    print("Decrypted messages:", [hex(m) for m in decrypted_messages])


    # # 128-bit messages
    # messages = [(1 << 126) + 13, (1 << 125) + 27, (1 << 124) + 57]    
    # print("Originaltexts:", messages)

    # ciphertexts = rsa.encrypt(messages, e, n)
    # print("Ciphertexts:", ciphertexts)

    # decrypted_messages = rsa.decrypt(ciphertexts, d, n)
    # print("Decrypted messages:", decrypted_messages)


# if __name__ == "__main__":
#     rsa = ParallelRSA2048()

#     # Example 2048-bit RSA parameters (truncated for simplicity)
#     n = 0xDCC897FFEE00123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567
#     e = 65537
#     d = 0x1E998A89CDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01

#     messages = [
#         0x123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234
#     ]

#     ciphertexts = rsa.encrypt(messages, e, n)
#     print("Ciphertexts:", ciphertexts)

#     # decrypted_messages = rsa.decrypt(ciphertexts, d, n)
#     # print("Decrypted messages:", decrypted_messages)
