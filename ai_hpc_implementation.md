# AI/HPC Systems Engineering Implementation Guide

## Environment Configuration

### Base System Setup
```bash
# Essential development tools
sudo apt update && sudo apt install -y build-essential git cmake gcc g++ gfortran
sudo apt install -y libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev

# NVIDIA drivers and CUDA toolkit
nvidia-smi  # Check current driver
sudo apt install -y nvidia-cuda-toolkit
nvcc --version
```

### Container Environment
```bash
# Docker installation
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt update && sudo apt install -y docker-ce
sudo usermod -aG docker $USER

# Apptainer/Singularity for HPC
sudo apt install -y apptainer

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Job Scheduling Implementation

### SLURM Configuration
```bash
# Install SLURM
sudo apt install -y slurmd slurmctld

# Basic configuration for local testing
sudo cat > /etc/slurm/slurm.conf << 'EOF'
ClusterName=localcluster
SlurmctldHost=localhost
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=1
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
SlurmUser=slurm
StateSaveLocation=/var/spool/slurmctld
SwitchType=switch/none
TaskPlugin=task/affinity
FastSchedule=1
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU_Memory,CR_CORE
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/none
SlurmctldLogFile=/var/log/slurmctld.log
SlurmdLogFile=/var/log/slurmd.log
NodeName=localhost CPUs=$(nproc) RealMemory=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}') State=UNKNOWN
PartitionName=debug Nodes=localhost Default=YES MaxTime=INFINITE State=UP
EOF

# Enable and start services
sudo systemctl enable slurmd slurmctld
sudo systemctl start slurmd slurmctld
```

### Example Job Templates

#### Basic CPU Job
```bash
#!/bin/bash
#SBATCH --job-name=cpu_job
#SBATCH --output=cpu_job_%j.out
#SBATCH --error=cpu_job_%j.err
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --mem=4G

echo "Running on $(hostname)"
echo "Job started at $(date)"

# Your application here
sleep 60

echo "Job finished at $(date)"
```

#### GPU Job
```bash
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --output=gpu_job_%j.out
#SBATCH --error=gpu_job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G

echo "Running on $(hostname)"
echo "Job started at $(date)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

nvidia-smi

# Your GPU application here
python3 /path/to/gpu_script.py

echo "Job finished at $(date)"
```

## AI Computing Framework

### PyTorch Environment Setup
```bash
# Create virtual environment
python3 -m venv ~/ai-env
source ~/ai-env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
  print('Device count:', torch.cuda.device_count()); \
  print('Device name:', torch.cuda.get_device_name(0))"

# Additional ML libraries
pip install numpy pandas scikit-learn matplotlib jupyter tensorboard mlflow
```

### Distributed Training Implementation

```python
# distributed_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x

def setup(rank, world_size):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed process group"""
    dist.destroy_process_group()

def train(rank, world_size, epochs=10):
    # Initialize process group
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = CustomModel(input_dim=784, hidden_dim=256, output_dim=10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Sample dataset (replace with actual data)
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for data shuffling
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            
            # Forward pass
            output = ddp_model(data)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
    
    # Save model (only on rank 0)
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "model.pt")
    
    cleanup()

if __name__ == "__main__":
    # Get world size from environment variable or use GPU count
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)
```

## Data Pipeline Implementation

### Parallel Data Processing
```python
# parallel_data_processing.py
import dask.dataframe as dd
import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster

def setup_dask_cluster(n_workers=4):
    """Setup local Dask cluster"""
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=2)
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    return client

def process_large_dataset(input_path, output_path):
    """Process large dataset in parallel"""
    # Initialize cluster
    client = setup_dask_cluster()
    
    # Read data
    ddf = dd.read_csv(input_path, blocksize="64MB")
    
    # Apply transformations
    ddf = ddf.map_partitions(clean_data)
    ddf = ddf.map_partitions(feature_engineering)
    
    # Compute results
    print("Computing results...")
    ddf = ddf.compute()
    
    # Save results
    ddf.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Close client
    client.close()

def clean_data(df):
    """Clean data partition"""
    # Drop nulls or fill with appropriate values
    df = df.dropna(subset=['critical_column'])
    df = df.fillna(0)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle outliers
    for col in ['col1', 'col2']:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    
    return df

def feature_engineering(df):
    """Feature engineering for partition"""
    # Create new features
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day'] = pd.to_datetime(df['date']).dt.day
    
    # Normalize numerical features
    for col in ['num_feature1', 'num_feature2']:
        if col in df.columns:
            df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

if __name__ == "__main__":
    input_path = "path/to/large/dataset.csv"
    output_path = "path/to/output.csv"
    process_large_dataset(input_path, output_path)
```

## Container-Based ML Environment

### Docker ML Development Container
```dockerfile
# Dockerfile.ml
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    libopenmpi-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Create conda environment
RUN conda create -y -n ml python=3.10 && \
    echo "source activate ml" >> ~/.bashrc

# Activate environment and install packages
SHELL ["/bin/bash", "-c"]
RUN source activate ml && \
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda install -y jupyterlab matplotlib pandas scikit-learn && \
    pip install dask distributed mlflow ray tensorboard

# Set working directory
WORKDIR /workspace

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Launch Jupyter Lab when container starts
CMD ["bash", "-c", "source activate ml && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
```

### Build and Run Container
```bash
# Build the container
docker build -t ml-dev-env -f Dockerfile.ml .

# Run container with GPU access
docker run --gpus all -it --rm -p 8888:8888 -p 6006:6006 -v $(pwd):/workspace ml-dev-env
```

## Performance Monitoring

### GPU Monitoring Setup
```bash
# Install monitoring tools
sudo apt install -y prometheus-node-exporter prometheus grafana

# Create NVIDIA GPU metrics exporter
cat > nvidia-gpu-exporter.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

def get_gpu_metrics():
    """Get NVIDIA GPU metrics using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        metrics = []
        for line in result.stdout.strip().split('\n'):
            if line:
                values = [val.strip() for val in line.split(',')]
                idx, name, temp, util_gpu, util_mem, mem_total, mem_used, mem_free, power = values
                
                metrics.append(f'# HELP gpu_{idx}_temperature GPU Temperature in Celsius')
                metrics.append(f'# TYPE gpu_{idx}_temperature gauge')
                metrics.append(f'gpu_{idx}_temperature {temp}')
                
                metrics.append(f'# HELP gpu_{idx}_utilization GPU Utilization in %')
                metrics.append(f'# TYPE gpu_{idx}_utilization gauge')
                metrics.append(f'gpu_{idx}_utilization {util_gpu}')
                
                metrics.append(f'# HELP gpu_{idx}_memory_utilization GPU Memory Utilization in %')
                metrics.append(f'# TYPE gpu_{idx}_memory_utilization gauge')
                metrics.append(f'gpu_{idx}_memory_utilization {util_mem}')
                
                metrics.append(f'# HELP gpu_{idx}_memory_used GPU Memory Used in MiB')
                metrics.append(f'# TYPE gpu_{idx}_memory_used gauge')
                metrics.append(f'gpu_{idx}_memory_used {mem_used}')
                
                metrics.append(f'# HELP gpu_{idx}_memory_total GPU Memory Total in MiB')
                metrics.append(f'# TYPE gpu_{idx}_memory_total gauge')
                metrics.append(f'gpu_{idx}_memory_total {mem_total}')
                
                metrics.append(f'# HELP gpu_{idx}_power_draw GPU Power Draw in Watts')
                metrics.append(f'# TYPE gpu_{idx}_power_draw gauge')
                metrics.append(f'gpu_{idx}_power_draw {power}')
        
        return '\n'.join(metrics)
    except Exception as e:
        return f"# ERROR: {str(e)}"

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            metrics = get_gpu_metrics()
            self.wfile.write(metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

if __name__ == '__main__':
    server_address = ('', 9100)
    httpd = HTTPServer(server_address, MetricsHandler)
    print(f'Starting NVIDIA GPU metrics exporter on port {server_address[1]}...')
    httpd.serve_forever()
EOF

chmod +x nvidia-gpu-exporter.py

# Create systemd service for exporter
sudo cat > /etc/systemd/system/nvidia-gpu-exporter.service << 'EOF'
[Unit]
Description=NVIDIA GPU Metrics Exporter
After=network.target

[Service]
ExecStart=/usr/bin/python3 /path/to/nvidia-gpu-exporter.py
Restart=always
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia-gpu-exporter
sudo systemctl start nvidia-gpu-exporter
```

## Security Implementation

### Secure AI Environment
```bash
# Create secure configuration for compute environment
cat > secure_ai_config.sh << 'EOF'
#!/bin/bash

# 1. Setup restricted user for AI workloads
sudo useradd -m -s /bin/bash aiuser
sudo usermod -aG docker aiuser

# 2. Configure Docker security
sudo cat > /etc/docker/daemon.json << 'DOCKERCONF'
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "userns-remap": "aiuser",
  "no-new-privileges": true,
  "live-restore": true,
  "userland-proxy": false,
  "seccomp-profile": "/etc/docker/seccomp-profile.json"
}
DOCKERCONF

# 3. Setup network isolation
sudo iptables -A INPUT -p tcp -m tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp -m tcp --dport 8888 -j ACCEPT
sudo iptables -A INPUT -p tcp -m tcp --dport 6006 -j ACCEPT
sudo iptables -A INPUT -p tcp -m tcp --dport 9100 -j ACCEPT
sudo iptables -A INPUT -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A INPUT -j DROP

# 4. Setup storage encryption
sudo apt install -y cryptsetup
sudo dd if=/dev/zero of=/ai_data.img bs=1M count=10240
sudo cryptsetup -v luksFormat /ai_data.img
sudo cryptsetup luksOpen /ai_data.img ai_data
sudo mkfs.ext4 /dev/mapper/ai_data
sudo mkdir -p /ai_data
sudo mount /dev/mapper/ai_data /ai_data
sudo chown -R aiuser:aiuser /ai_data

# 5. Setup automatic updates
sudo apt install -y unattended-upgrades apt-listchanges
sudo cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'UPDATES'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}";
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::Package-Blacklist {
};
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "02:00";
UPDATES

# Enable unattended upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
EOF

chmod +x secure_ai_config.sh
```

## Production Deployment Template

### Model Serving with Kubernetes
```yaml
# ai-deployment.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-serving

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  namespace: ai-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: model-serving
        image: your-registry/model-serving:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
        ports:
        - containerPort: 8080
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 15
        volumeMounts:
        - name: model-storage
          mountPath: /models
        env:
        - name: MODEL_PATH
          value: "/models/model.pt"
        - name: WORKERS
          value: "4"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: ai-serving
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: model-serving
  namespace: ai-serving
spec:
  selector:
    app: model-serving
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## CI/CD Pipeline Configuration

### GitHub Actions for AI Development
```yaml
# .github/workflows/ai-pipeline.yml
name: AI/ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
  
  build-container:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_HUB_USERNAME }}
        password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./Dockerfile.ml
        push: true
        tags: ${{ secrets.DOCKER_HUB_USERNAME }}/ai-model:latest

  deploy:
    needs: build-container
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Install kubectl
      uses: azure/setup-kubectl@v3
    - name: Configure kubeconfig
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" > kubeconfig.yaml
        export KUBECONFIG=kubeconfig.yaml
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/ai-deployment.yaml
        kubectl rollout restart deployment/model-serving -n ai-serving
```

## Future Development Roadmap

1. **Implement Automated Model Selection Framework**
   - Integration with hyperparameter optimization libraries
   - A/B testing capabilities for model deployment

2. **Enhanced Resource Management**
   - Dynamic resource allocation based on workload demands
   - Predictive scaling for ML workloads

3. **Integration with MLOps Platforms**
   - Model versioning and lineage tracking
   - Experiment management and reproducibility

4. **Federated Learning Support**
   - Secure multi-node training capabilities
   - Cross-organizational model training

5. **Quantum Computing Integration**
   - Hybrid classical-quantum computing workflows
   - QISKIT or Pennylane integration
