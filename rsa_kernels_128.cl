// Function declarations
void multiply_128(ulong a_low, ulong a_high, ulong b_low, ulong b_high, __private ulong *result_low, __private ulong *result_high);
void mod_128(ulong a_low, ulong a_high, ulong mod_low, ulong mod_high, __private ulong *result_low, __private ulong *result_high);

__kernel void mod_exp_128(__global const ulong *messages_low, 
                          __global const ulong *messages_high, 
                          ulong exp, 
                          __global const ulong *mod_low, 
                          __global const ulong *mod_high, 
                          __global ulong *result_low, 
                          __global ulong *result_high, 
                          uint count) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    // Load inputs
    ulong m_low = messages_low[gid];
    ulong m_high = messages_high[gid];
    ulong mod_l = mod_low[0];
    ulong mod_h = mod_high[0];

    // Initialize result to 1 (128-bit)
    ulong r_low = 1;
    ulong r_high = 0;

    // Base and temporary storage
    ulong base_low = m_low;
    ulong base_high = m_high;

    // Temporary variables for intermediate results
    ulong temp_low, temp_high;

    // Modular exponentiation
    while (exp > 0) {
        if (exp & 1) {  // Multiply result = result * base % mod
            multiply_128(r_low, r_high, base_low, base_high, &temp_low, &temp_high);
            mod_128(temp_low, temp_high, mod_l, mod_h, &r_low, &r_high);
        }
        // Square the base
        multiply_128(base_low, base_high, base_low, base_high, &temp_low, &temp_high);
        mod_128(temp_low, temp_high, mod_l, mod_h, &base_low, &base_high);

        exp >>= 1;
    }

    result_low[gid] = r_low;
    result_high[gid] = r_high;
}

// Multiply two 128-bit numbers represented as (low, high) pairs
void multiply_128(ulong a_low, ulong a_high, ulong b_low, ulong b_high, __private ulong *result_low, __private ulong *result_high) {
    ulong low = a_low * b_low;
    ulong mid1 = a_high * b_low;
    ulong mid2 = a_low * b_high;
    ulong high = a_high * b_high;

    ulong carry = (mid1 + mid2) >> 64;
    *result_low = low + ((mid1 & 0xFFFFFFFFFFFFFFFF) << 64);
    *result_high = high + (mid1 >> 64) + (mid2 >> 64) + carry;
}

// Modular reduction for a 128-bit number
void mod_128(ulong a_low, ulong a_high, ulong mod_low, ulong mod_high, __private ulong *result_low, __private ulong *result_high) {
    // Simplified modulus operation for 128-bit numbers
    while (a_high > mod_high || (a_high == mod_high && a_low >= mod_low)) {
        if (a_low < mod_low) {
            a_low += (1UL << 64) - mod_low;
            a_high -= 1;
        } else {
            a_low -= mod_low;
        }
        a_high -= mod_high;
    }
    *result_low = a_low;
    *result_high = a_high;
}
