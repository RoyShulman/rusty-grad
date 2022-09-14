use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const TRAINING_SET_IMAGES: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
const TRAINING_SET_LABELS: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";

const TEST_SET_IMAGES: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
const TEST_SET_LABELS: &str =
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz";

pub struct DatasetPath {
    pub images: PathBuf,
    pub labels: PathBuf,
}

fn download_file(url: &str, directory: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let filename = url.split("/").last().ok_or("Could not split url")?;
    let filepath = directory.join(filename);
    if !filepath.exists() {
        let resp = reqwest::blocking::get(url)?.bytes()?;
        fs::write(&filepath, resp)?;
    }
    Ok(filepath)
}

pub fn download_dataset(dir: &Path) -> Result<DatasetPath, Box<dyn Error>> {
    match fs::create_dir(dir) {
        Ok(_) => (),
        Err(err) => match err.kind() {
            std::io::ErrorKind::AlreadyExists => (),
            _ => return Err(Box::new(err)),
        },
    };

    let images = download_file(TRAINING_SET_IMAGES, dir)?;
    let labels = download_file(TRAINING_SET_LABELS, dir)?;
    Ok(DatasetPath { images, labels })
}

pub fn download_testset(dir: &Path) -> Result<DatasetPath, Box<dyn Error>> {
    match fs::create_dir(dir) {
        Ok(_) => (),
        Err(err) => match err.kind() {
            std::io::ErrorKind::AlreadyExists => (),
            _ => return Err(Box::new(err)),
        },
    };

    let images = download_file(TEST_SET_IMAGES, dir)?;
    let labels = download_file(TEST_SET_LABELS, dir)?;
    Ok(DatasetPath { images, labels })
}
