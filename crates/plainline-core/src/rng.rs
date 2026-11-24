//! NumPy-compatible random numbers: `SeedSequence`, `PCG64` and `Generator.integers(0, n, size)`.
//!
//! The bootstrap draws item indices exactly as `numpy.random.default_rng(seed).integers(0, n, n)`
//! does, so a Rust bootstrap and the Python version it replaced agree draw for draw. The
//! tests check this against sequences recorded from NumPy.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL: usize = 4;

const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// Split a non-negative integer seed into 32-bit words, least significant first.
fn seed_words(mut seed: u128) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    while seed > 0 {
        out.push((seed & 0xffff_ffff) as u32);
        seed >>= 32;
    }
    out
}

/// `numpy.random.SeedSequence(seed).generate_state(n_words, np.uint64)`.
pub fn seed_sequence_state(seed: u128, n_words: usize) -> Vec<u64> {
    let entropy = seed_words(seed);
    let mut hash_const = INIT_A;
    let mut pool = [0u32; POOL];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for i_src in 0..POOL {
        for i_dst in 0..POOL {
            if i_src != i_dst {
                let h = hashmix(pool[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
    }
    for &e in entropy.iter().skip(POOL) {
        for slot in pool.iter_mut() {
            let h = hashmix(e, &mut hash_const);
            *slot = mix(*slot, h);
        }
    }
    let mut hash_const = INIT_B;
    let words32: Vec<u32> = (0..n_words * 2)
        .map(|i| {
            let mut v = pool[i % POOL] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^ (v >> XSHIFT)
        })
        .collect();
    words32
        .chunks(2)
        .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
        .collect()
}

/// The PCG64 bit generator as NumPy seeds and steps it, with its 32-bit output buffer.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    /// `numpy.random.PCG64(seed)`.
    pub fn new(seed: u128) -> Self {
        let v = seed_sequence_state(seed, 4);
        let initstate = (u128::from(v[0]) << 64) | u128::from(v[1]);
        let initseq = (u128::from(v[2]) << 64) | u128::from(v[3]);
        let mut g = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        let rot = (s >> 122) as u32;
        (((s >> 64) as u64) ^ (s as u64)).rotate_right(rot)
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }

    /// One draw of `Generator.integers(0, n)` (int64) for 0 < n <= 2^32 (Lemire's method).
    pub fn bounded(&mut self, n: u64) -> u64 {
        assert!((1..=(1u64 << 32)).contains(&n), "range must fit 32 bits");
        let rng = (n - 1) as u32;
        if rng == 0 {
            return 0;
        }
        if rng == u32::MAX {
            return u64::from(self.next_u32());
        }
        let rng_excl = u64::from(rng) + 1;
        let mut m = u64::from(self.next_u32()) * rng_excl;
        let mut leftover = m & 0xffff_ffff;
        if leftover < rng_excl {
            let threshold = u64::from(u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u64::from(self.next_u32()) * rng_excl;
                leftover = m & 0xffff_ffff;
            }
        }
        m >> 32
    }
}

/// `numpy.random.default_rng(seed)`, enough of it for the bootstrap.
#[derive(Debug, Clone)]
pub struct Generator(Pcg64);

impl Generator {
    pub fn new(seed: u64) -> Self {
        Generator(Pcg64::new(u128::from(seed)))
    }

    /// Fill `out` like `rng.integers(0, n, len(out))`.
    pub fn fill_integers(&mut self, n: usize, out: &mut [usize]) {
        for slot in out.iter_mut() {
            *slot = self.0.bounded(n as u64) as usize;
        }
    }
}
