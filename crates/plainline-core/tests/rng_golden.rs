//! The bootstrap RNG must reproduce NumPy's `default_rng(seed).integers(0, n, size)` exactly.
//! `tests/golden/rng.json` was recorded from NumPy 2.4.6.

use plainline_core::rng::Generator;
use serde_json::Value;

#[test]
fn matches_numpy_integers() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/rng.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut checked = 0;
    for case in cases.as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let mut g = Generator::new(seed);
        for call in case["calls"].as_array().unwrap() {
            let n = call["n"].as_u64().unwrap() as usize;
            let size = call["size"].as_u64().unwrap() as usize;
            let want: Vec<usize> = call["draws"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let mut got = vec![0usize; size];
            g.fill_integers(n, &mut got);
            assert_eq!(got, want, "seed {seed} n {n}");
            checked += 1;
        }
    }
    assert!(checked > 50);
}
