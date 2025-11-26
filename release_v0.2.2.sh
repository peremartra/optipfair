#!/bin/bash

# OptiPFair v0.2.2 Release Script
# This script helps automate the release process

set -e  # Exit on error

echo "=========================================="
echo "OptiPFair v0.2.2 Release Process"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Verify we're on main branch
echo -e "${YELLOW}Step 1: Verifying branch...${NC}"
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}Error: Not on main branch. Current branch: $CURRENT_BRANCH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ On main branch${NC}"
echo ""

# Step 2: Check for uncommitted changes
echo -e "${YELLOW}Step 2: Checking for uncommitted changes...${NC}"
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Uncommitted changes found. Showing status:${NC}"
    git status -s
    echo ""
    read -p "Do you want to commit these changes? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Release v0.2.2: Selective layer width pruning & optimized hybrid importance"
        echo -e "${GREEN}✓ Changes committed${NC}"
    else
        echo -e "${RED}Aborting release. Please commit or stash changes first.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ No uncommitted changes${NC}"
fi
echo ""

# Step 3: Run tests
echo -e "${YELLOW}Step 3: Running tests...${NC}"
read -p "Do you want to run tests before releasing? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "pytest" ] || command -v pytest &> /dev/null; then
        python -m pytest tests/test_selective_layer_pruning.py -v
        echo -e "${GREEN}✓ Tests passed${NC}"
    else
        echo -e "${YELLOW}pytest not found. Skipping tests.${NC}"
    fi
fi
echo ""

# Step 4: Build package
echo -e "${YELLOW}Step 4: Building package...${NC}"
read -p "Build package for PyPI? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Clean old builds
    rm -rf dist/ build/ *.egg-info
    echo "Cleaning old builds..."
    
    # Build
    python setup.py sdist bdist_wheel
    echo -e "${GREEN}✓ Package built${NC}"
    echo ""
    echo "Built files:"
    ls -lh dist/
else
    echo -e "${YELLOW}Skipping package build${NC}"
fi
echo ""

# Step 5: Create git tag
echo -e "${YELLOW}Step 5: Creating git tag...${NC}"
if git rev-parse v0.2.2 >/dev/null 2>&1; then
    echo -e "${YELLOW}Tag v0.2.2 already exists${NC}"
    read -p "Do you want to delete and recreate it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -d v0.2.2
        git push origin :refs/tags/v0.2.2 2>/dev/null || true
        git tag -a v0.2.2 -m "Version 0.2.2 - Selective Layer Width Pruning"
        echo -e "${GREEN}✓ Tag recreated${NC}"
    fi
else
    git tag -a v0.2.2 -m "Version 0.2.2 - Selective Layer Width Pruning"
    echo -e "${GREEN}✓ Tag created${NC}"
fi
echo ""

# Step 6: Push to GitHub
echo -e "${YELLOW}Step 6: Pushing to GitHub...${NC}"
read -p "Push to GitHub? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    git push origin v0.2.2
    echo -e "${GREEN}✓ Pushed to GitHub${NC}"
else
    echo -e "${YELLOW}Skipping GitHub push${NC}"
fi
echo ""

# Step 7: Upload to PyPI
echo -e "${YELLOW}Step 7: Uploading to PyPI...${NC}"
read -p "Upload to PyPI? (requires twine and credentials) (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v twine &> /dev/null; then
        echo "Uploading to PyPI..."
        twine upload dist/*
        echo -e "${GREEN}✓ Uploaded to PyPI${NC}"
    else
        echo -e "${RED}Error: twine not found. Install with: pip install twine${NC}"
    fi
else
    echo -e "${YELLOW}Skipping PyPI upload${NC}"
    echo ""
    echo "To upload manually later:"
    echo "  twine upload dist/*"
fi
echo ""

# Step 8: Create GitHub Release
echo -e "${YELLOW}Step 8: GitHub Release${NC}"
echo ""
echo "To create the GitHub release:"
echo "1. Go to: https://github.com/peremartra/optipfair/releases/new"
echo "2. Select tag: v0.2.2"
echo "3. Title: OptiPFair v0.2.2 - Selective Layer Width Pruning"
echo "4. Copy content from: RELEASE_NOTES_v0.2.2.md"
echo "5. Attach files from: dist/"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Release Process Complete!${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Version: 0.2.2"
echo "  Tag: v0.2.2"
echo "  Release Notes: RELEASE_NOTES_v0.2.2.md"
echo ""
echo "Next steps:"
echo "  1. Create GitHub Release (see instructions above)"
echo "  2. Verify PyPI page: https://pypi.org/project/optipfair/"
echo "  3. Test installation: pip install optipfair==0.2.2"
echo "  4. Announce on social media/forums"
echo ""
echo -e "${GREEN}🎉 Congratulations on the release!${NC}"
