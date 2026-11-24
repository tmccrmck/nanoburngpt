use crate::data::{TextDataset, load_text, load_text_file};
use std::{
    fmt, fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

const WIKITEXT103_URL: &str =
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip";

#[derive(Debug)]
pub enum Dataset {
    Shakespeare,
    WikiText103,
}

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shakespeare => write!(f, "shakespeare"),
            Self::WikiText103 => write!(f, "wikitext103"),
        }
    }
}

impl std::str::FromStr for Dataset {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "shakespeare" => Ok(Self::Shakespeare),
            "wikitext103" => Ok(Self::WikiText103),
            other => anyhow::bail!(
                "Unknown dataset '{}'. Available: shakespeare, wikitext103",
                other
            ),
        }
    }
}

impl Dataset {
    /// Path where the preprocessed training plain-text file is cached locally.
    pub fn local_path(&self) -> PathBuf {
        match self {
            Self::Shakespeare => PathBuf::from("data/shakespeare/input.txt"),
            Self::WikiText103 => PathBuf::from("data/wikitext103/train.txt"),
        }
    }

    /// Optional canonical validation split path (WikiText-103 has `wiki.valid.raw`).
    pub fn valid_path(&self) -> Option<PathBuf> {
        match self {
            Self::Shakespeare => None,
            Self::WikiText103 => Some(PathBuf::from("data/wikitext103/valid.txt")),
        }
    }

    /// Ensure the dataset is downloaded and preprocessed. Returns the local train path.
    pub fn ensure_downloaded(&self) -> anyhow::Result<PathBuf> {
        let path = self.local_path();
        let valid_ready = self.valid_path().as_ref().is_none_or(|p| p.exists());
        if path.exists() && valid_ready {
            return Ok(path);
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        match self {
            Self::Shakespeare => download_shakespeare(&path),
            Self::WikiText103 => {
                let valid = self.valid_path().expect("wikitext103 has valid path");
                download_wikitext103(&path, &valid)
            }
        }?;

        Ok(path)
    }

    /// Download (if needed) and load `(train, val)` datasets using canonical splits when available.
    pub fn load(&self, block_size: usize) -> anyhow::Result<(TextDataset, TextDataset)> {
        let train_path = self.ensure_downloaded()?;
        match self.valid_path() {
            Some(valid_path) => Ok((
                load_text_file(&train_path, block_size)?,
                load_text_file(&valid_path, block_size)?,
            )),
            None => load_text(&train_path, block_size),
        }
    }
}

fn download_shakespeare(dest: &Path) -> anyhow::Result<()> {
    println!("Downloading Shakespeare from {}...", SHAKESPEARE_URL);
    let text = ureq::get(SHAKESPEARE_URL)
        .call()?
        .body_mut()
        .with_config()
        .limit(16 * 1024 * 1024)
        .read_to_string()?;
    fs::write(dest, text)?;
    println!("Saved to {}", dest.display());
    Ok(())
}

fn extract_zip_entry(archive: &mut zip::ZipArchive<Cursor<Vec<u8>>>, name: &str) -> anyhow::Result<String> {
    let result: anyhow::Result<String> = {
        match archive.by_name(name) {
            Err(_) => Err(anyhow::anyhow!("not found")),
            Ok(mut file) => {
                let mut buf = String::new();
                file.read_to_string(&mut buf)?;
                Ok(buf)
            }
        }
    };
    match result {
        Ok(text) => Ok(text),
        Err(_) => {
            let names: Vec<String> = (0..archive.len())
                .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
                .collect();
            anyhow::bail!(
                "Could not find '{}' in zip. Available files: {:?}",
                name,
                names
            );
        }
    }
}

fn download_wikitext103(train_dest: &Path, valid_dest: &Path) -> anyhow::Result<()> {
    println!("Downloading WikiText-103 from {}...", WIKITEXT103_URL);
    let bytes = ureq::get(WIKITEXT103_URL)
        .call()?
        .body_mut()
        .with_config()
        .limit(512 * 1024 * 1024)
        .read_to_vec()?;
    println!("Extracting zip ({} MB)...", bytes.len() / 1_000_000);

    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let train_raw = extract_zip_entry(&mut archive, "wikitext-103-raw/wiki.train.raw")?;
    let valid_raw = extract_zip_entry(&mut archive, "wikitext-103-raw/wiki.valid.raw")?;

    fs::write(train_dest, strip_wikitext_markup(&train_raw))?;
    fs::write(valid_dest, strip_wikitext_markup(&valid_raw))?;
    println!(
        "Saved preprocessed train/valid splits to {} and {}",
        train_dest.display(),
        valid_dest.display()
    );
    Ok(())
}

fn strip_wikitext_markup(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with('=') && trimmed.ends_with('='))
        })
        .map(|line| {
            line.replace(" @-@ ", "-")
                .replace(" @.@ ", ".")
                .replace(" @,@ ", ",")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn dataset_from_str_shakespeare() {
        let d = Dataset::from_str("shakespeare").unwrap();
        assert!(matches!(d, Dataset::Shakespeare));
    }

    #[test]
    fn dataset_from_str_wikitext103() {
        let d = Dataset::from_str("wikitext103").unwrap();
        assert!(matches!(d, Dataset::WikiText103));
    }

    #[test]
    fn dataset_from_str_unknown_errors() {
        assert!(Dataset::from_str("imagenet").is_err());
    }

    #[test]
    fn strip_removes_heading_lines() {
        let input = "= Heading =\nSome text\n = Sub = \n more text";
        let out = strip_wikitext_markup(input);
        assert!(!out.contains("Heading"), "heading should be removed");
        assert!(out.contains("Some text"));
        assert!(out.contains("more text"));
    }

    #[test]
    fn strip_replaces_at_dash_artifacts() {
        let input = "2000 @-@ 2001 season";
        let out = strip_wikitext_markup(input);
        assert_eq!(out, "2000-2001 season");
    }

    #[test]
    fn strip_replaces_at_dot_and_comma_artifacts() {
        let input = "U @.@ S @.@ A @,@ which";
        let out = strip_wikitext_markup(input);
        assert_eq!(out, "U.S.A,which");
    }

    #[test]
    fn strip_preserves_non_heading_equals() {
        // A line with = in the middle but not wrapping both ends should survive.
        let input = "x = 5\nnormal text";
        let out = strip_wikitext_markup(input);
        assert!(out.contains("x = 5"));
    }
}
