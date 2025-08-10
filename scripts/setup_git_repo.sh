#!/bin/bash
# Setup script for initializing the SAC Trading Agent Git repository

set -e  # Exit on any error

echo "=========================================="
echo "SAC Trading Agent - Git Repository Setup"
echo "=========================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Create .gitkeep files for empty directories
echo "Creating .gitkeep files for empty directories..."
mkdir -p data models logs config
touch data/.gitkeep models/.gitkeep logs/.gitkeep config/.gitkeep

# Add all files to git
echo "Adding files to Git..."
git add .

# Check if there are any changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit"
else
    # Make initial commit
    echo "Making initial commit..."
    git commit -m "Initial commit: SAC Trading Agent

- Add SAC agent implementation with CNN-LSTM networks
- Add prioritized experience replay buffer
- Add trading environment simulation
- Add configuration management
- Add training and testing scripts
- Add comprehensive documentation and setup files"
    echo "✓ Initial commit completed"
fi

# Setup Git hooks (optional)
echo "Setting up Git hooks..."
mkdir -p .git/hooks

# Pre-commit hook for code formatting
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to run black formatter

echo "Running pre-commit checks..."

# Check if black is installed
if ! command -v black &> /dev/null; then
    echo "Warning: black is not installed. Please run 'pip install black' to enable code formatting."
    exit 0
fi

# Run black on Python files
echo "Running black formatter..."
black --check src/ scripts/ --exclude=__pycache__ 2>/dev/null || {
    echo "Code formatting issues found. Running black formatter..."
    black src/ scripts/ --exclude=__pycache__
    echo "Code formatted. Please add the changes and commit again."
    exit 1
}

echo "✓ Pre-commit checks passed"
EOF

chmod +x .git/hooks/pre-commit
echo "✓ Pre-commit hook installed"

# Setup branch protection (for when pushed to remote)
echo "Setting up recommended Git configuration..."
git config --local pull.rebase false  # Use merge strategy for pulls
git config --local init.defaultBranch main  # Use 'main' as default branch

# Check current branch name and rename to 'main' if needed
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ] && [ "$current_branch" == "master" ]; then
    echo "Renaming default branch from 'master' to 'main'..."
    git branch -m master main
    echo "✓ Default branch renamed to 'main'"
fi

echo ""
echo "=========================================="
echo "Git repository setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create a repository on GitHub/GitLab"
echo "2. Add remote origin:"
echo "   git remote add origin https://github.com/yourusername/sac-trading-agent.git"
echo "3. Push to remote:"
echo "   git push -u origin main"
echo ""
echo "Repository structure:"
git log --oneline -n 5 2>/dev/null || echo "No commits yet"
echo ""
echo "Files tracked:"
git ls-files | head -10
if [ $(git ls-files | wc -l) -gt 10 ]; then
    echo "... and $(( $(git ls-files | wc -l) - 10 )) more files"
fi
echo ""
echo "Happy coding! 🚀"