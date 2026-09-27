#!/usr/bin/env bash
set -euo pipefail
REGION=us-east-1
aws sts get-caller-identity
aws service-quotas list-service-quotas --service-code ec2 --region "$REGION" \
 --query "Quotas[contains(QuotaName, 'Running On-Demand G')].[QuotaName,Value,QuotaCode]" --output table
aws ec2 describe-instances --region "$REGION" \
 --filters Name=instance-state-name,Values=pending,running \
 --query 'Reservations[].Instances[].{Type:InstanceType,Id:InstanceId,State:State.Name}' --output table
aws ec2 describe-instance-types --region "$REGION" --instance-types g6.24xlarge g5.24xlarge g6e.12xlarge \
 --query 'InstanceTypes[].{Type:InstanceType,vCPU:VCpuInfo.DefaultVCpus,RAMMiB:MemoryInfo.SizeInMiB,GPU:GpuInfo.Gpus}' --output json
aws ec2 describe-instance-type-offerings --region "$REGION" --location-type availability-zone \
 --filters Name=instance-type,Values=g6.24xlarge,g5.24xlarge,g6e.12xlarge --output table
printf '\nOfferings are not live capacity guarantees. G quota must cover default vCPUs plus existing use.\n'
aws ssm get-parameter --region "$REGION" \
 --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
 --query Parameter.Value --output text
