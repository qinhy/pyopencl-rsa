import numpy as np

class BigNum:
    def __init__(self, value=0):
        # value can be an integer or a numpy array of uint64
        if isinstance(value, int):
            # Convert integer to array of uint64
            self.digits = np.zeros(32, dtype=np.uint64)
            temp_value = value
            for i in range(32):
                self.digits[i] = temp_value & 0xFFFFFFFFFFFFFFFF
                temp_value >>= 64
                if temp_value == 0:
                    break
        elif isinstance(value, np.ndarray):
            if value.dtype != np.uint64 or value.size != 32:
                raise ValueError("Value must be a numpy array of 32 uint64 elements")
            self.digits = value.copy()
        else:
            raise TypeError("Unsupported type for BigNum initialization")

    def copy(self):
        return BigNum(self.digits.copy())

    def __add__(self, other):
        # Addition of two BigNums
        result = BigNum()
        carry = np.uint64(0)
        for i in range(32):
            total = np.uint64(self.digits[i]) + np.uint64(other.digits[i]) + carry
            result.digits[i] = total & np.uint64(0xFFFFFFFFFFFFFFFF)
            carry = total >> np.uint64(64)
        return result

    def __sub__(self, other):
        # Subtraction of two BigNums
        result = BigNum()
        borrow = np.uint64(0)
        for i in range(32):
            diff = np.uint64(self.digits[i]) - np.uint64(other.digits[i]) - borrow
            if self.digits[i] < other.digits[i] + borrow:
                borrow = np.uint64(1)
                diff += np.uint64(1) << np.uint64(64)
            else:
                borrow = np.uint64(0)
            result.digits[i] = diff & np.uint64(0xFFFFFFFFFFFFFFFF)
        return result

    def __mul__(self, other):
        # Multiplication of two BigNums
        result = BigNum()
        temp = np.zeros(64, dtype=np.uint64)
        for i in range(32):
            carry = np.uint64(0)
            for j in range(32):
                if i + j < 64:
                    a = np.uint64(self.digits[i])
                    b = np.uint64(other.digits[j])
                    total = np.uint64(temp[i + j]) + a * b + carry
                    temp[i + j] = total & np.uint64(0xFFFFFFFFFFFFFFFF)
                    carry = total >> np.uint64(64)
            if i + 32 < 64:
                temp[i + 32] += carry
        result.digits = temp[:32]
        return result

    def __eq__(self, other):
        return np.array_equal(self.digits, other.digits)

    def __lt__(self, other):
        for i in reversed(range(32)):
            if self.digits[i] < other.digits[i]:
                return True
            elif self.digits[i] > other.digits[i]:
                return False
        return False

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __lshift__(self, n):
        if n < 0:
            return self.__rshift__(-n)
        result = BigNum()
        word_shift = n // 64
        bit_shift = n % 64
        for i in reversed(range(32)):
            if i - word_shift >= 0:
                result.digits[i] = self.digits[i - word_shift] << bit_shift
                if bit_shift != 0 and i - word_shift - 1 >= 0:
                    result.digits[i] |= self.digits[i - word_shift - 1] >> (64 - bit_shift)
                result.digits[i] &= np.uint64(0xFFFFFFFFFFFFFFFF)
        return result

    def __rshift__(self, n):
        if n < 0:
            return self.__lshift__(-n)
        result = BigNum()
        word_shift = n // 64
        bit_shift = n % 64
        for i in range(32):
            if i + word_shift < 32:
                result.digits[i] = self.digits[i + word_shift] >> bit_shift
                if bit_shift != 0 and i + word_shift + 1 < 32:
                    result.digits[i] |= self.digits[i + word_shift + 1] << (64 - bit_shift)
                result.digits[i] &= np.uint64(0xFFFFFFFFFFFFFFFF)
        return result

    def __mod__(self, modulus):
        # Modulo operation using division
        _, remainder = self.__divmod__(modulus)
        return remainder

    def __floordiv__(self, other):
        quotient, _ = self.__divmod__(other)
        return quotient

    def __divmod__(self, other):
        if other == BigNum(0):
            raise ZeroDivisionError("division by zero")
        quotient = BigNum(0)
        remainder = BigNum(0)
        for i in reversed(range(32 * 64)):
            # Shift remainder left by 1 bit and add the next bit from self
            remainder = remainder << 1
            bit = (self.digits[i // 64] >> (i % 64)) & np.uint64(1)
            remainder.digits[0] |= bit
            # If remainder >= other, subtract other from remainder and set quotient bit
            if remainder >= other:
                remainder = remainder - other
                quotient.digits[i // 64] |= np.uint64(1) << (i % 64)
        return quotient, remainder

    def __pow__(self, exponent, modulus):
        # Modular exponentiation using square-and-multiply
        result = BigNum(1)
        base = self % modulus
        for i in range(32 * 64 - 1, -1, -1):
            result = (result * result) % modulus
            bit = (exponent.digits[i // 64] >> (i % 64)) & np.uint64(1)
            if bit == 1:
                result = (result * base) % modulus
            print(result)
        return result

    def __repr__(self):
        # Convert BigNum to hex string
        hex_str = ''.join('{:016x}'.format(d) for d in reversed(self.digits)).lstrip('0') or '0'
        return '0x' + hex_str

    def __str__(self):
        return self.__repr__()

# Example usage:

# Define modulus n, public exponent e, private exponent d
# For testing purposes, we use small numbers, in practice these should be 2048-bit numbers
n = BigNum(0xD5BBB96D30086EC484EBA3D7F9CAEB07)
e = BigNum(3)
d = BigNum(0x9A1CFC78B8E1A1C9C2F5D0A2E6E4F5F1)

# Message to encrypt (must be less than n)
message = BigNum(123456789)

# Encrypt
ciphertext = message.__pow__(e, n)

print("Ciphertext:", ciphertext)

# Decrypt
decrypted = ciphertext.__pow__(d, n)

print("Decrypted:", decrypted)

# Verify
if decrypted == message:
    print("Success: Decrypted message matches original message.")
else:
    print("Failure: Decrypted message does not match original message.")
