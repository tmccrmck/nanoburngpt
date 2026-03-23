use std::{
    fmt,
    fs,
    io::{Cursor, Read},
    path::PathBuf,
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
    /// Path where the preprocessed plain-text file is cached locally.
    pub fn local_path(&self) -> PathBuf {
        match self {
            Self::Shakespeare => PathBuf::from("data/shakespeare/input.txt"),
            Self::WikiText103 => PathBuf::from("data/wikitext103/input.txt"),
        }
    }

    /// Ensure the dataset is downloaded and preprocessed. Returns the local path.
    pub fn ensure_downloaded(&self) -> anyhow::Result<PathBuf> {
        let path = self.local_path();
        if path.exists() {
            return Ok(path);
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        match self {
            Self::Shakespeare => download_shakespeare(&path),
            Self::WikiText103 => download_wikitext103(&path),
        }?;

        Ok(path)
    }
}

fn download_shakespeare(dest: &std::path::Path) -> anyhow::Result<()> {
    println!("Downloading Shakespeare from {}...", SHAKESPEARE_URL);
    let text = reqwest::blocking::get(SHAKESPEARE_URL)?.text()?;
    fs::write(dest, text)?;
    println!("Saved to {}", dest.display());
    Ok(())
}

fn download_wikitext103(dest: &std::path::Path) -> anyhow::Result<()> {
    println!("Downloading WikiText-103 from {}...", WIKITEXT103_URL);
    let bytes = reqwest::blocking::get(WIKITEXT103_URL)?.bytes()?;
    println!("Extracting zip ({} MB)...", bytes.len() / 1_000_000);

    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let target_files = [
        "wikitext-103-raw/wiki.train.raw",
        "wikitext-103-raw/wiki.valid.raw",
    ];

    let mut combined = String::new();
    for name in &target_files {
        // Read the file content into a local buffer, fully releasing the borrow on `archive`
        // before we might need it again to list available entries on error.
        let result: anyhow::Result<String> = {
            let found = archive.by_name(name);
            match found {
                Err(_) => Err(anyhow::anyhow!("not found")),
                Ok(mut file) => {
                    let mut buf = String::new();
                    file.read_to_string(&mut buf)?;
                    Ok(buf)
                }
            }
        };
        match result {
            Ok(text) => {
                combined.push_str(&text);
                combined.push('\n');
            }
            Err(_) => {
                let names: Vec<String> = (0..archive.len())
                    .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
                    .collect();
                anyhow::bail!(
                    "Could not find '{}' in zip. Available files: {:?}",
                    name, names
                );
            }
        }
    }

    let cleaned = strip_wikitext_markup(&combined);
    fs::write(dest, cleaned)?;
    println!("Saved preprocessed text to {}", dest.display());
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
