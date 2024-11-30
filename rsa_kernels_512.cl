#define CHUNK_SIZE 8 // 512 bits = 8 * 64-bit chunks

void multiply_512(ulong *a, ulong *b, ulong *result) {
    ulong temp[2 * CHUNK_SIZE] = {0};  // Temporary array for 1024-bit result

    for (int i = 0; i < CHUNK_SIZE; i++) {
        for (int j = 0; j < CHUNK_SIZE; j++) {
            ulong prod = a[i] * b[j];
            ulong carry = 0;
            int k = i + j;

            // Add product to temporary array
            do {
                prod += temp[k] + carry;
                temp[k] = prod & 0xFFFFFFFFFFFFFFFF;
                carry = prod >> 64;
                k++;
            } while (carry != 0);
        }
    }

    // Copy lower 512 bits to result
    for (int i = 0; i < CHUNK_SIZE; i++) {
        result[i] = temp[i];
    }
}

void subtract_512(ulong *a, ulong *b, ulong *result) {
    ulong borrow = 0;

    for (int i = 0; i < CHUNK_SIZE; i++) {
        ulong diff = a[i] - b[i] - borrow;

        // Check for borrow
        borrow = (a[i] < (b[i] + borrow)) ? 1 : 0;

        result[i] = diff;
    }
}

// Helper function to check if the exponent is zero
int is_zero(ulong *exp) {
    for (int i = 0; i < CHUNK_SIZE; i++) {
        if (exp[i] != 0) return 0;
    }
    return 1;
}

// Helper function to right-shift the exponent by 1 (exp >>= 1)
void shift_right(ulong *exp) {
    for (int i = 0; i < CHUNK_SIZE; i++) {
        ulong next = (i + 1 < CHUNK_SIZE) ? exp[i + 1] : 0;
        exp[i] = (exp[i] >> 1) | (next << 63);
    }
}

// Modular exponentiation: base^exp % mod
__kernel void mod_exp_512(__global const ulong *numbers, 
                          __global const ulong *exponents,
                          __global const ulong *mod,
                          __global ulong *results,
                          uint count) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    ulong base[CHUNK_SIZE], exp[CHUNK_SIZE], mod_val[CHUNK_SIZE];
    ulong result[CHUNK_SIZE], temp[CHUNK_SIZE];
    
    // Initialize arrays
    for (int i = 0; i < CHUNK_SIZE; i++) {
        base[i] = numbers[gid * CHUNK_SIZE + i];
        exp[i] = exponents[gid * CHUNK_SIZE + i];
        mod_val[i] = mod[i];
        result[i] = (i == 0) ? 1 : 0;  // Initialize result to 1
    }

    while (!is_zero(exp)) {
        // If the least significant bit of the exponent is 1
        if (exp[0] & 1) {
            multiply_512(result, base, temp);
            subtract_512(temp, mod_val, result); // Reduce modulo
        }

        multiply_512(base, base, temp);
        subtract_512(temp, mod_val, base); // Reduce modulo

        shift_right(exp); // exp >>= 1
    }

    for (int i = 0; i < CHUNK_SIZE; i++) {
        results[gid * CHUNK_SIZE + i] = result[i];
    }
}
