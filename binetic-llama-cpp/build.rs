//! Build script for binetic-llama-cpp — compiles llama.cpp as a static library.

fn main() {
    let llama_cpp_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("backends").join("llama-cpp");

    // Collect .cpp files from llama.cpp src/ (library core only)
    let mut cpp_files = Vec::new();
    let skip_cpp = ["models", "cli", "server", "common"];
    scan_cpp(&llama_cpp_dir.join("src"), &mut cpp_files, &skip_cpp);

    // GGML core .c files
    let ggml_c = [
        llama_cpp_dir.join("ggml/src/ggml.c"),
        llama_cpp_dir.join("ggml/src/ggml-quants.c"),
    ];

    cc::Build::new()
        .cpp_link_stdlib("c++")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-DGGML_STATIC_DEFINE")
        .flag("-DLLAMA_BUILD")
        .flag("-DGGML_VERSION=0")
        .flag("-DGGML_COMMIT=\"\"")
        .flag("-D_GNU_SOURCE")
        .include(&llama_cpp_dir)
        .include(llama_cpp_dir.join("include"))
        .include(llama_cpp_dir.join("ggml/include"))
        .include(llama_cpp_dir.join("ggml/src"))
        .include(llama_cpp_dir.join("ggml/src/ggml-cpu"))
        .files(&ggml_c)
        .files(&cpp_files)
        .compile("llama-cpp");

    println!("cargo:link-lib=c++");
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        llama_cpp_dir.join("include/llama.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        llama_cpp_dir.join("ggml/include/ggml.h").display()
    );
}

fn scan_cpp(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>, skip: &[&str]) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if skip.contains(&name) {
                    continue;
                }
                scan_cpp(&path, out, skip);
            } else if path.extension().map_or(false, |e| e == "cpp") {
                let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if fname != "llama.cpp" && fname != "llama-model.cpp" {
                    out.push(path);
                }
            }
        }
    }
}
