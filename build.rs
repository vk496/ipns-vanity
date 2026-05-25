// On Linux distributions that ship only `libOpenCL.so.1` (no `-dev` package),
// the linker can't resolve `-lOpenCL`. Drop a versioned symlink into OUT_DIR
// and put OUT_DIR on the library search path.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if !cfg!(target_os = "linux") {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let link = out_dir.join("libOpenCL.so");

    if !link.exists() {
        let candidates = [
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
            "/usr/lib64/libOpenCL.so.1",
            "/usr/lib/libOpenCL.so.1",
        ];
        for cand in &candidates {
            if std::path::Path::new(cand).exists()
                && std::os::unix::fs::symlink(cand, &link).is_ok()
            {
                break;
            }
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
