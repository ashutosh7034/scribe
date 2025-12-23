# Scribe - Real-Time Sign Language Translation Platform

Scribe is a generative AI platform that provides real-time conversion of spoken language into expressive sign language using a customizable 3D avatar.

## Architecture

- **Backend**: Python FastAPI with AWS services integration
- **Frontend**: React with TypeScript and WebGL for 3D avatar rendering
- **Infrastructure**: AWS services deployed via CDK
- **Development**: Docker containers for local development

## Quick Start

1. **Prerequisites**:
   - Docker and Docker Compose
   - Node.js 18+ and Python 3.11+
   - AWS CLI configured

2. **Development Setup**:
   ```bash
   # Start all services
   docker-compose up -d
   
   # Backend will be available at http://localhost:8000
   # Frontend will be available at http://localhost:3000
   ```

3. **Infrastructure Deployment**:
   ```bash
   cd infrastructure
   npm install
   cdk deploy
   ```

## Project Structure

```
scribe/
├── backend/           # FastAPI Python backend
├── frontend/          # React TypeScript frontend
├── infrastructure/    # AWS CDK infrastructure code
├── docker-compose.yml # Development environment
└── README.md
```

## Requirements

This project implements requirements 6.1 (multi-platform accessibility) and 8.1 (web interface) from the Scribe specification.