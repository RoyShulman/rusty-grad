use core::fmt;
use std::{
    error::{self, Error},
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use flate2::read::GzDecoder;

use crate::download_utils::DatasetPath;

#[derive(Debug)]
enum DatasetFileError {
    InvalidHeaderMagic(u32),
}

impl fmt::Display for DatasetFileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Both underlying errors already impl `Display`, so we defer to
            // their implementations.
            DatasetFileError::InvalidHeaderMagic(magic) => write!(f, "Invalid magic: {}", magic),
        }
    }
}

impl error::Error for DatasetFileError {}

fn read_u32_be<R: BufRead>(reader: &mut R) -> Result<u32, Box<dyn Error>> {
    let mut buf = vec![0u8; 4];
    reader.read_exact(&mut buf)?;
    let mut result: u32 = *buf.get(3).ok_or("read_u32_be - At index 3")? as u32;
    result += (*buf.get(2).ok_or("read_u32_be - At index 2")? as u32) << 8;
    result += (*buf.get(1).ok_or("read_u32_be - At index 1")? as u32) << 16;
    result += (*buf.get(0).ok_or("read_u32_be - At index 0")? as u32) << 24;

    Ok(result)
}

const IMAGE_FILE_MAGIC: u32 = 2051;
const LABEL_FILE_MAGIC: u32 = 2049;

pub struct Image {
    pub pixels: Vec<u8>,
}

pub struct LabeledImage {
    pub pixels: Image,
    pub label: u8,
}

fn get_and_validate_data_file(
    path: &Path,
    expected_magic: u32,
) -> Result<BufReader<GzDecoder<File>>, Box<dyn Error>> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(GzDecoder::new(f));

    let magic = read_u32_be(&mut reader)?;
    if magic != expected_magic {
        return Err(Box::new(DatasetFileError::InvalidHeaderMagic(magic)));
    }

    Ok(reader)
}

fn get_training_images(path: &Path) -> Result<Vec<Image>, Box<dyn Error>> {
    let mut reader = get_and_validate_data_file(path, IMAGE_FILE_MAGIC)?;

    let num_items = read_u32_be(&mut reader)?;
    let rows = read_u32_be(&mut reader)?;
    let columns = read_u32_be(&mut reader)?;

    let mut images = vec![];
    for _ in 0..num_items {
        let mut pixels = vec![0u8; (rows * columns) as usize];
        reader.read_exact(&mut pixels)?;
        images.push(Image { pixels })
    }
    Ok(images)
}

fn get_training_labels(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut reader = get_and_validate_data_file(path, LABEL_FILE_MAGIC)?;

    let num_items = read_u32_be(&mut reader)?;
    let mut labels = vec![0u8; num_items as usize];
    reader.read_exact(&mut labels)?;
    Ok(labels)
}

pub fn get_training_set(dataset_path: &DatasetPath) -> Result<Vec<LabeledImage>, Box<dyn Error>> {
    let pixels = get_training_images(&dataset_path.images)?;
    let labels = get_training_labels(&dataset_path.labels)?;

    let mut training_set = vec![];
    for (image, label) in pixels.into_iter().zip(labels) {
        training_set.push(LabeledImage {
            pixels: image,
            label,
        });
    }
    Ok(training_set)
}
