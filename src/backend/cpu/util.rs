use std::fs::File;
use std::io::{self, BufRead, BufReader};


pub(crate) fn get_cpu_description() -> io::Result<String> {
    let file = File::open("/proc/cpuinfo")?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("model name") {
            if let Some((_, value)) = line.split_once(':') {
                return Ok(value.trim().to_string());
            }
        }
    }

    Ok("CPU".to_string())
}
