#!/bin/bash
# Rollback for Base System Configuration
# Reverts changes made by setup_environment.sh

# Log all output
exec > >(tee -i base_system_rollback.log)
exec 2>&1

echo "Starting rollback of base system configuration"
echo "Timestamp: $(date)"

# Create backup of bashrc before modification
cp ~/.bashrc ~/.bashrc.bak

# Remove CUDA path additions from bashrc
sed -i '/export PATH=\/usr\/local\/cuda\/bin:\$PATH/d' ~/.bashrc
sed -i '/export LD_LIBRARY_PATH=\/usr\/local\/cuda\/lib64:\$LD_LIBRARY_PATH/d' ~/.bashrc

echo "Removed CUDA path configurations from ~/.bashrc"

# List packages to remove (in reverse order of installation)
PACKAGES_TO_REMOVE=(
  "nvidia-cuda-toolkit"
  "nvidia-driver"
  "python3-venv"
  "python3-pip" 
  "python3-dev"
  "liblapack-dev"
  "libopenblas-dev"
  "libopenmpi-dev"
  "gfortran"
  "automake"
  "autoconf"
  "cmake"
  "git"
  "build-essential"
)

echo "The following packages will be removed:"
printf '%s\n' "${PACKAGES_TO_REMOVE[@]}"
read -p "Proceed with package removal? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  for package in "${PACKAGES_TO_REMOVE[@]}"; do
    echo "Removing $package..."
    apt-get remove -y "$package"
  done
  
  echo "Running autoremove to clean up dependencies..."
  apt-get autoremove -y
else
  echo "Package removal skipped"
fi

echo "Rollback completed"
echo "Timestamp: $(date)"
echo "Note: System updates and upgrades have not been rolled back"
