fn main() {
    let args: Vec<String> = std::env::args().collect();
    std::process::exit(plainline_cli::run_cli(&args));
}
