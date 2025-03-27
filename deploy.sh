#!/bin/bash
# Deployment script for Helios RF Suite

set -e

# Configuration
VERSION="0.1.0"
PACKAGE_NAME="helios-rf"
DIST_DIR="./dist"
CONFIG_DIR="./config"
DATA_DIR="./data"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Helios RF Suite Deployment Script${NC}"
echo "Version: $VERSION"

# Create virtual environment
echo -e "\n${GREEN}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Install build dependencies
echo -e "\n${GREEN}Installing build dependencies...${NC}"
pip install --upgrade pip
pip install build twine wheel

# Clean previous builds
echo -e "\n${GREEN}Cleaning previous builds...${NC}"
rm -rf $DIST_DIR
rm -rf build
rm -rf *.egg-info

# Build package
echo -e "\n${GREEN}Building package...${NC}"
python -m build

# Verify the build
echo -e "\n${GREEN}Verifying build...${NC}"
twine check $DIST_DIR/*

# Create deployment package
echo -e "\n${GREEN}Creating deployment package...${NC}"
DEPLOY_DIR="deploy_$PACKAGE_NAME-$VERSION"
mkdir -p $DEPLOY_DIR

# Copy distribution files
cp $DIST_DIR/* $DEPLOY_DIR/

# Copy configuration files
mkdir -p $DEPLOY_DIR/config
cp $CONFIG_DIR/*.yaml $DEPLOY_DIR/config/
cp $CONFIG_DIR/*.json $DEPLOY_DIR/config/

# Copy sample data
mkdir -p $DEPLOY_DIR/data
cp -r $DATA_DIR/models $DEPLOY_DIR/data/
cp -r $DATA_DIR/scenarios $DEPLOY_DIR/data/

# Copy Docker files
cp Dockerfile $DEPLOY_DIR/
cp docker-compose.yml $DEPLOY_DIR/ 2>/dev/null || echo "No docker-compose.yml found"

# Copy documentation
cp README.md $DEPLOY_DIR/
cp LICENSE $DEPLOY_DIR/ 2>/dev/null || echo "No LICENSE file found"

# Create installation script
cat > $DEPLOY_DIR/install.sh << 'EOF'
#!/bin/bash
# Helios RF Suite Installation Script

set -e

# Check Python version
python3 --version

# Create virtual environment
python3 -m venv helios-env
source helios-env/bin/activate

# Install the package
pip install --upgrade pip
pip install helios-rf-*.whl

# Create data directories
mkdir -p ~/helios-data/results

echo "Installation complete!"
echo "Activate the environment with: source helios-env/bin/activate"
echo "Run the simulator with: helios-sim -c config/default_sim_config.yaml"
EOF

chmod +x $DEPLOY_DIR/install.sh

# Create a zip archive
echo -e "\n${GREEN}Creating zip archive...${NC}"
zip -r "${DEPLOY_DIR}.zip" $DEPLOY_DIR

echo -e "\n${GREEN}Deployment package created: ${DEPLOY_DIR}.zip${NC}"
echo "You can distribute this package for installation."

# Cleanup
rm -rf $DEPLOY_DIR

# Deactivate virtual environment
deactivate

echo -e "\n${GREEN}Deployment completed successfully!${NC}"