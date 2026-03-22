# spillover project justfile
# All repeating commands should be recipes here.
# Agents MUST read and use these recipes.

# Default recipe: show available commands
default:
    @just --list

# choose recipes interactively
choose:
    @just --choose

# === Development Workflow ===

# Run all pre-commit checks (required before committing)
check: fmt-check lint test doc-check
    @echo "All checks passed"

# Run checks on all files (required before pushing)
check-all: fmt-check lint-all test-all doc-check
    @echo "All checks passed on full codebase"

# === Formatting ===

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Apply formatting fixes
fmt:
    cargo fmt --all

# === Linting ===

# Run clippy with deny warnings (on changed files via cargo check)
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Run clippy on all files
lint-all:
    cargo clippy --all-targets --all-features -- -D warnings

# === Testing ===

# Run tests with nextest (--no-tests=pass allows empty test suites)
test:
    cargo nextest run --all-features --no-tests=pass

# Run all tests including ignored
test-all:
    cargo nextest run --all-features --run-ignored all --no-tests=pass

# Run tests with verbose output
test-verbose:
    cargo nextest run --all-features --no-capture --no-tests=pass

# === Building ===

# Build debug binary
build:
    cargo build

# Build release binary
build-release:
    cargo build --release

# Check compilation without building
check-compile:
    cargo check --all-targets --all-features

# === jj Workflow ===

# Prepare a commit: run all checks, then show status
prepare-commit: check
    @echo ""
    @echo "Ready to commit. Run: jj commit -m 'your message'"
    @jj status

# Prepare for push: run full checks
prepare-push: check-all
    @echo ""
    @echo "Ready to push. Run: jj git push"

# Show current jj status
status:
    jj status

# Show jj log
log:
    jj log

# === Utility ===

# Clean build artifacts
clean:
    cargo clean

# Update dependencies
update:
    cargo update

# === Documentation ===

# Check that documentation builds without errors
doc-check:
    cargo doc --no-deps --document-private-items

# Generate and open documentation
doc:
    cargo doc --no-deps --open

# Count source lines of code (excluding blanks and comments)
sloc:
    @tokei --types=Rust --compact

# === Project Setup ===

# Full project setup for new clones (run this first!)
setup: clone-refs
    @echo ""
    @echo "Project setup complete!"
    @echo "Reference repos: .agents/repos/"

# === Reference Repositories ===

# Clone reference repositories as shallow clones via jj (--force to overwrite)
[arg("force", long="force", value="1")]
clone-refs force="":
    @echo "Cloning reference repositories into .agents/repos/..."
    @{{ if force != "" { "rm -rf .agents/repos" } else { "true" } }}
    @mkdir -p .agents/repos
    @echo "Cloning sra-taxa-rs (reference sort implementation)..."
    jj git clone --depth 1 https://github.com/nrminor/sra-taxa-rs.git .agents/repos/sra-taxa-rs || echo "sra-taxa-rs already exists, skipping"
    @echo "Cloning dryice (temporary on-disk format)..."
    jj git clone --depth 1 https://github.com/nrminor/dryice.git .agents/repos/dryice || echo "dryice already exists, skipping"
    @echo "Cloning samtools (reference: BAM sorting)..."
    jj git clone --depth 1 https://github.com/samtools/samtools.git .agents/repos/samtools || echo "samtools already exists, skipping"
    @echo "Cloning fastq-tools (reference: FASTQ utilities)..."
    jj git clone --depth 1 https://github.com/dcjones/fastq-tools.git .agents/repos/fastq-tools || echo "fastq-tools already exists, skipping"
    @echo "Cloning seqkit (reference: sequence toolkit)..."
    jj git clone --depth 1 https://github.com/shenwei356/seqkit.git .agents/repos/seqkit || echo "seqkit already exists, skipping"
    @echo "Cloning haveitnway (prototype sort implementation)..."
    jj git clone --depth 1 https://github.com/nrminor/haveitnway.git .agents/repos/haveitnway || echo "haveitnway already exists, skipping"
    @echo "Reference repositories cloned to .agents/repos/"

# Remove reference repositories
clean-refs:
    @echo "Removing reference repositories..."
    rm -rf .agents/repos
    @echo "Reference repositories removed"
