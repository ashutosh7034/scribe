/**
 * Scribe Infrastructure Stack
 * 
 * AWS CDK stack definition for the Scribe platform infrastructure.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

interface ScribeInfrastructureStackProps extends cdk.StackProps {
  environment: string;
}

export class ScribeInfrastructureStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: ScribeInfrastructureStackProps) {
    super(scope, id, props);

    const { environment } = props;

    // VPC for all resources
    const vpc = new ec2.Vpc(this, 'ScribeVPC', {
      maxAzs: 3,
      natGateways: environment === 'production' ? 3 : 1,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
        {
          cidrMask: 28,
          name: 'Database',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
        },
      ],
    });

    // S3 Bucket for avatar assets and model storage
    const assetsBucket = new s3.Bucket(this, 'ScribeAssetsBucket', {
      bucketName: `scribe-assets-${environment}-${this.account}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          id: 'DeleteOldVersions',
          enabled: true,
          noncurrentVersionExpiration: cdk.Duration.days(30),
        },
      ],
    });

    // CloudFront distribution for asset delivery
    const distribution = new cloudfront.Distribution(this, 'ScribeAssetsDistribution', {
      defaultBehavior: {
        origin: new origins.S3Origin(assetsBucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
      },
      priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
    });

    // RDS PostgreSQL Database
    const database = new rds.DatabaseInstance(this, 'ScribeDatabase', {
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.VER_15,
      }),
      instanceType: environment === 'production' 
        ? ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM)
        : ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
      vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      },
      multiAz: environment === 'production',
      allocatedStorage: environment === 'production' ? 100 : 20,
      storageEncrypted: true,
      backupRetention: cdk.Duration.days(environment === 'production' ? 7 : 1),
      deletionProtection: environment === 'production',
      databaseName: 'scribe',
      credentials: rds.Credentials.fromGeneratedSecret('scribe-admin'),
    });

    // ElastiCache Redis Cluster
    const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
      description: 'Subnet group for Redis cluster',
      subnetIds: vpc.privateSubnets.map(subnet => subnet.subnetId),
    });

    const redisSecurityGroup = new ec2.SecurityGroup(this, 'RedisSecurityGroup', {
      vpc,
      description: 'Security group for Redis cluster',
      allowAllOutbound: false,
    });

    const redis = new elasticache.CfnCacheCluster(this, 'ScribeRedisCluster', {
      cacheNodeType: environment === 'production' ? 'cache.t3.medium' : 'cache.t3.micro',
      engine: 'redis',
      numCacheNodes: 1,
      vpcSecurityGroupIds: [redisSecurityGroup.securityGroupId],
      cacheSubnetGroupName: redisSubnetGroup.ref,
    });

    // ECS Cluster
    const cluster = new ecs.Cluster(this, 'ScribeCluster', {
      vpc,
      containerInsights: true,
    });

    // Application Load Balancer
    const alb = new elbv2.ApplicationLoadBalancer(this, 'ScribeALB', {
      vpc,
      internetFacing: true,
      securityGroup: new ec2.SecurityGroup(this, 'ALBSecurityGroup', {
        vpc,
        description: 'Security group for Application Load Balancer',
        allowAllOutbound: true,
      }),
    });

    // Allow HTTP and HTTPS traffic to ALB
    alb.connections.allowFromAnyIpv4(ec2.Port.tcp(80), 'HTTP traffic');
    alb.connections.allowFromAnyIpv4(ec2.Port.tcp(443), 'HTTPS traffic');

    // Task Definition for Backend Service
    const backendTaskDefinition = new ecs.FargateTaskDefinition(this, 'BackendTaskDefinition', {
      memoryLimitMiB: environment === 'production' ? 2048 : 512,
      cpu: environment === 'production' ? 1024 : 256,
    });

    // Backend Container
    const backendContainer = backendTaskDefinition.addContainer('backend', {
      image: ecs.ContainerImage.fromRegistry('scribe/backend:latest'), // TODO: Replace with actual image
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'backend',
        logGroup: new logs.LogGroup(this, 'BackendLogGroup', {
          logGroupName: `/ecs/scribe-backend-${environment}`,
          retention: logs.RetentionDays.ONE_WEEK,
        }),
      }),
      environment: {
        ENVIRONMENT: environment,
        AWS_REGION: this.region,
        REDIS_URL: `redis://${redis.attrRedisEndpointAddress}:${redis.attrRedisEndpointPort}`,
      },
      secrets: {
        DATABASE_URL: ecs.Secret.fromSecretsManager(database.secret!, 'engine'),
      },
    });

    backendContainer.addPortMappings({
      containerPort: 8000,
      protocol: ecs.Protocol.TCP,
    });

    // ECS Service for Backend
    const backendService = new ecs.FargateService(this, 'BackendService', {
      cluster,
      taskDefinition: backendTaskDefinition,
      desiredCount: environment === 'production' ? 2 : 1,
      assignPublicIp: false,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      },
    });

    // Allow backend to connect to database
    database.connections.allowFrom(backendService, ec2.Port.tcp(5432));

    // Allow backend to connect to Redis
    redisSecurityGroup.addIngressRule(
      ec2.Peer.securityGroupId(backendService.connections.securityGroups[0].securityGroupId),
      ec2.Port.tcp(6379),
      'Allow backend to connect to Redis'
    );

    // Target Group for Backend Service
    const backendTargetGroup = new elbv2.ApplicationTargetGroup(this, 'BackendTargetGroup', {
      vpc,
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      healthCheck: {
        path: '/health',
        healthyHttpCodes: '200',
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3,
      },
    });

    backendTargetGroup.addTarget(backendService);

    // ALB Listener
    const listener = alb.addListener('Listener', {
      port: 80,
      defaultTargetGroups: [backendTargetGroup],
    });

    // API Gateway for WebSocket connections
    const webSocketApi = new apigateway.RestApi(this, 'ScribeWebSocketAPI', {
      restApiName: `scribe-websocket-${environment}`,
      description: 'WebSocket API for real-time translation',
      deployOptions: {
        stageName: environment,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: true,
      },
    });

    // IAM Role for ECS tasks to access AWS services
    const taskRole = new iam.Role(this, 'ECSTaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
      inlinePolicies: {
        S3Access: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                's3:GetObject',
                's3:PutObject',
                's3:DeleteObject',
              ],
              resources: [
                assetsBucket.bucketArn,
                `${assetsBucket.bucketArn}/*`,
              ],
            }),
          ],
        }),
        TranscribeAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                'transcribe:StartStreamTranscription',
                'transcribe:StartTranscriptionJob',
              ],
              resources: ['*'],
            }),
          ],
        }),
      },
    });

    backendTaskDefinition.addToTaskRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          's3:GetObject',
          's3:PutObject',
          'transcribe:StartStreamTranscription',
        ],
        resources: ['*'],
      })
    );

    // Outputs
    new cdk.CfnOutput(this, 'LoadBalancerDNS', {
      value: alb.loadBalancerDnsName,
      description: 'DNS name of the load balancer',
    });

    new cdk.CfnOutput(this, 'DatabaseEndpoint', {
      value: database.instanceEndpoint.hostname,
      description: 'RDS database endpoint',
    });

    new cdk.CfnOutput(this, 'RedisEndpoint', {
      value: redis.attrRedisEndpointAddress,
      description: 'Redis cluster endpoint',
    });

    new cdk.CfnOutput(this, 'AssetsBucketName', {
      value: assetsBucket.bucketName,
      description: 'S3 bucket for assets',
    });

    new cdk.CfnOutput(this, 'CloudFrontDistributionDomain', {
      value: distribution.distributionDomainName,
      description: 'CloudFront distribution domain',
    });

    new cdk.CfnOutput(this, 'WebSocketAPIEndpoint', {
      value: webSocketApi.url,
      description: 'WebSocket API endpoint',
    });
  }
}