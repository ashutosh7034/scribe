#!/usr/bin/env node
/**
 * Scribe Infrastructure CDK App
 * 
 * Main entry point for AWS CDK infrastructure deployment.
 */

import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { ScribeInfrastructureStack } from '../lib/scribe-infrastructure-stack';

const app = new cdk.App();

// Get environment configuration
const account = process.env.CDK_DEFAULT_ACCOUNT;
const region = process.env.CDK_DEFAULT_REGION || 'us-east-1';
const environment = process.env.ENVIRONMENT || 'development';

new ScribeInfrastructureStack(app, `ScribeInfrastructureStack-${environment}`, {
  env: { 
    account, 
    region 
  },
  environment,
  
  // Stack tags
  tags: {
    Project: 'Scribe',
    Environment: environment,
    ManagedBy: 'CDK',
  },
});