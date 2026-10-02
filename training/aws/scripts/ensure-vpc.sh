#!/usr/bin/env bash
# Idempotent: a VPC of the project's own for regions with no default VPC. One public
# subnet per AZ in $AZS (instances get a public IP for SSM, S3 and Hugging Face egress; the
# security group admits nothing), an internet gateway and a route table, all tagged
# Name=$HOBSON_NAME_TAG. Prints the VPC id; put it in HOBSON_VPC (local.env) so ensure_sg
# and launch-host.sh use it instead of the default VPC.
# Usage: AZS="us-east-1a us-east-1c" training/aws/scripts/ensure-vpc.sh <region> [cidr /16, default 10.77.0.0/16]
set -euo pipefail
source "$(dirname "$0")/common.sh"
R=${1:?region}; CIDR=${2:-10.77.0.0/16}; AZS=${AZS:?set AZS to the availability zones to cover}
TAGS="Tags=[{Key=Name,Value=$HOBSON_NAME_TAG},{Key=hobson:job-type,Value=$HOBSON_JOB_TAG}]"
ec2() { aws ec2 --region "$R" "$@"; }

vpc=$(ec2 describe-vpcs --filters "Name=tag:Name,Values=$HOBSON_NAME_TAG" --query 'Vpcs[0].VpcId' --output text)
if [[ "$vpc" == None ]]; then
  vpc=$(ec2 create-vpc --cidr-block "$CIDR" --tag-specifications "ResourceType=vpc,$TAGS" --query Vpc.VpcId --output text)
  ec2 wait vpc-available --vpc-ids "$vpc"
  ec2 modify-vpc-attribute --vpc-id "$vpc" --enable-dns-hostnames
  log "created VPC $vpc ($CIDR)"
fi
igw=$(ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$vpc" --query 'InternetGateways[0].InternetGatewayId' --output text)
if [[ "$igw" == None ]]; then
  igw=$(ec2 create-internet-gateway --tag-specifications "ResourceType=internet-gateway,$TAGS" --query InternetGateway.InternetGatewayId --output text)
  ec2 attach-internet-gateway --internet-gateway-id "$igw" --vpc-id "$vpc"
  log "created and attached $igw"
fi
rt=$(ec2 describe-route-tables --filters "Name=vpc-id,Values=$vpc" "Name=tag:Name,Values=$HOBSON_NAME_TAG" --query 'RouteTables[0].RouteTableId' --output text)
if [[ "$rt" == None ]]; then
  rt=$(ec2 create-route-table --vpc-id "$vpc" --tag-specifications "ResourceType=route-table,$TAGS" --query RouteTable.RouteTableId --output text)
  ec2 create-route --route-table-id "$rt" --destination-cidr-block 0.0.0.0/0 --gateway-id "$igw" >/dev/null
  log "created route table $rt (0.0.0.0/0 -> $igw)"
fi
base=${CIDR%.*.*}; i=0
for az in $AZS; do
  i=$((i + 1))
  sn=$(ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" "Name=availability-zone,Values=$az" --query 'Subnets[0].SubnetId' --output text)
  if [[ "$sn" == None ]]; then
    sn=$(ec2 create-subnet --vpc-id "$vpc" --availability-zone "$az" --cidr-block "$base.$i.0/24" \
      --tag-specifications "ResourceType=subnet,$TAGS" --query Subnet.SubnetId --output text)
    ec2 modify-subnet-attribute --subnet-id "$sn" --map-public-ip-on-launch
    ec2 associate-route-table --route-table-id "$rt" --subnet-id "$sn" >/dev/null
    log "created subnet $sn in $az ($base.$i.0/24)"
  fi
done
echo "$vpc"
