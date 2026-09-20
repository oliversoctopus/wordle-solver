//! numpy's `default_rng(seed).choice(n, size, replace=True, p=weights)`.
//!
//! Training answers are drawn from the prior, and a fit is only comparable
//! with the Python's if the two drew the same answers. That means numpy's
//! generator rather than any generator: SeedSequence to spread the seed over a
//! pool, PCG64 to turn that into doubles, and `choice`'s own cumulative-sum and
//! right-hand binary search to turn doubles into words. All three are fixed by
//! numpy and none of them is a detail this can pick for itself.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

const PCG_MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^= value >> XSHIFT;
    value
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

/// The entropy pool `SeedSequence(seed)` builds for an integer seed.
fn seed_pool(seed: u64) -> [u32; POOL_SIZE] {
    // _coerce_to_uint32_array: an integer becomes its little-endian 32-bit
    // words, and zero becomes a single zero word rather than none at all.
    let mut entropy: Vec<u32> = Vec::new();
    let mut rest = seed;
    while rest > 0 {
        entropy.push((rest & 0xffff_ffff) as u32);
        rest >>= 32;
    }
    if entropy.is_empty() {
        entropy.push(0);
    }

    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                let mixed = hashmix(pool[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], mixed);
            }
        }
    }
    for &word in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            let mixed = hashmix(word, &mut hash_const);
            *slot = mix(*slot, mixed);
        }
    }
    pool
}

/// `SeedSequence.generate_state(n, uint64)`: 2n 32-bit words, paired up.
fn generate_state(pool: &[u32; POOL_SIZE], n_words: usize) -> Vec<u64> {
    let mut words = Vec::with_capacity(n_words * 2);
    let mut hash_const = INIT_B;
    for i in 0..(n_words * 2) {
        let mut value = pool[i % POOL_SIZE];
        value ^= hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        value = value.wrapping_mul(hash_const);
        value ^= value >> XSHIFT;
        words.push(value);
    }
    words
        .chunks(2)
        .map(|pair| (pair[1] as u64) << 32 | pair[0] as u64)
        .collect()
}

/// PCG64 XSL-RR, seeded the way `np.random.default_rng` seeds it.
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    pub fn new(seed: u64) -> Pcg64 {
        let words = generate_state(&seed_pool(seed), 4);
        let initstate = ((words[0] as u128) << 64) | words[1] as u128;
        let initseq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut rng = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let state = self.state;
        let xored = ((state >> 64) ^ state) as u64;
        xored.rotate_right((state >> 122) as u32)
    }

    /// `Generator.random()`: the top 53 bits scaled into `[0, 1)`.
    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

/// `rng.choice(len(weights), size, replace=True, p=weights)`.
///
/// The cumulative sum runs left to right because `np.cumsum` does; using a
/// pairwise reduction here would be more accurate and would draw different
/// answers, which is the wrong trade in the one place that has to match.
pub fn sample_answers(weights: &[f64], size: usize, seed: u64) -> Vec<usize> {
    let mut cdf = Vec::with_capacity(weights.len());
    let mut running = 0.0;
    for &w in weights {
        running += w;
        cdf.push(running);
    }
    let total = *cdf.last().unwrap_or(&1.0);
    for slot in cdf.iter_mut() {
        *slot /= total;
    }

    let mut rng = Pcg64::new(seed);
    (0..size)
        .map(|_| {
            let u = rng.next_double();
            // searchsorted(cdf, u, side='right'): the first entry strictly above
            let mut low = 0usize;
            let mut high = cdf.len();
            while low < high {
                let mid = (low + high) / 2;
                if cdf[mid] <= u {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low.min(cdf.len() - 1)
        })
        .collect()
}
