use std::any::Any;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;

fn get_cpu_description() -> String {
    let file = File::open("/proc/cpuinfo").map_err("CPU")?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line.map_err(|_| "CPU")?;
        if line.starts_with("model name") {
            if let Some((_, value)) = line.split_once(':') {
                return value.trim().to_string();
            }
        }
    }
    "CPU"
}
