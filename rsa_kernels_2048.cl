#define N_LIMBS 32

typedef ulong limb_t;
typedef limb_t bigint_t[N_LIMBS];

void big_add(const bigint_t a, const bigint_t b, bigint_t result)
{
    ulong carry = 0;
    for(int i = 0; i < N_LIMBS; i++)
    {
        ulong temp = a[i] + b[i] + carry;
        if(temp < a[i])
            carry = 1;
        else if(temp == a[i] && carry == 1)
            carry = 1;
        else
            carry = 0;
        result[i] = temp;
    }
}

void big_sub(const bigint_t a, const bigint_t b, bigint_t result)
{
    ulong borrow = 0;
    for(int i = 0; i < N_LIMBS; i++)
    {
        ulong temp = a[i] - b[i] - borrow;
        if(a[i] < b[i] + borrow)
            borrow = 1;
        else
            borrow = 0;
        result[i] = temp;
    }
}

void big_mul(const bigint_t a, const bigint_t b, bigint_t result)
{
    // Initialize result to zero
    for(int i = 0; i < N_LIMBS * 2; i++)
        result[i] = 0;

    for(int i = 0; i < N_LIMBS; i++)
    {
        ulong carry = 0;
        for(int j = 0; j < N_LIMBS; j++)
        {
            ulong prod = (ulong)a[i] * (ulong)b[j];
            ulong temp = (ulong)result[i+j] + prod + carry;
            result[i+j] = (ulong)(temp & 0xFFFFFFFFFFFFFFFFUL);
            carry = (ulong)(temp >> 64);
        }
        result[i+N_LIMBS] += carry;
    }
}

int big_cmp(const bigint_t a, const bigint_t b)
{
    for(int i = N_LIMBS - 1; i >= 0; i--)
    {
        if(a[i] > b[i])
            return 1;
        else if(a[i] < b[i])
            return -1;
    }
    return 0;
}

void big_mod(bigint_t a, const bigint_t modulus, bigint_t result)
{
    // Simple modulo operation using subtraction
    // Not optimized for large numbers
    // Assumes a < modulus * 2
    bigint_t temp;
    for(int i = 0; i < N_LIMBS; i++)
        temp[i] = a[i];

    while(true)
    {
        int cmp = big_cmp(temp, modulus);
        if(cmp < 0)
            break;
        big_sub(temp, modulus, temp);
    }
    for(int i = 0; i < N_LIMBS; i++)
        result[i] = temp[i];
}

void big_copy(const bigint_t src, bigint_t dest)
{
    for(int i = 0; i < N_LIMBS; i++)
        dest[i] = src[i];
}

void big_mod_exp(const bigint_t base, const bigint_t exponent, const bigint_t modulus, bigint_t result)
{
    bigint_t base_mod;
    bigint_t exp;
    bigint_t temp_result;
    bigint_t temp_base;
    big_copy(base, temp_base);
    big_copy(exponent, exp);

    // Initialize result to 1
    for(int i = 0; i < N_LIMBS; i++)
        result[i] = 0;
    result[0] = 1;

    // While exponent > 0
    for(int i = N_LIMBS * 64 - 1; i >= 0; i--)
    {
        // result = (result * result) % modulus
        bigint_t res_square;
        big_mul(result, result, res_square);
        big_mod(res_square, modulus, result);

        // If exponent bit is 1
        ulong exp_word = exp[i / 64];
        if(exp_word & ((ulong)1 << (i % 64)))
        {
            // result = (result * base) % modulus
            bigint_t res_mul;
            big_mul(result, temp_base, res_mul);
            big_mod(res_mul, modulus, result);
        }
    }
}

__kernel void rsa_mod_exp(
    __global ulong* base,
    __global ulong* exponent,
    __global ulong* modulus,
    __global ulong* result,
    const uint num_messages)
{
    int gid = get_global_id(0);

    if(gid >= num_messages)
        return;

    // Load base, exponent, modulus into local variables
    bigint_t base_local;
    bigint_t exponent_local;
    bigint_t modulus_local;
    bigint_t result_local;

    for(int i = 0; i < N_LIMBS; i++)
    {
        base_local[i] = base[gid * N_LIMBS + i];
        exponent_local[i] = exponent[gid * N_LIMBS + i];
        modulus_local[i] = modulus[i];
    }

    big_mod_exp(base_local, exponent_local, modulus_local, result_local);

    // Store result
    for(int i = 0; i < N_LIMBS; i++)
    {
        result[gid * N_LIMBS + i] = result_local[i];
    }
}