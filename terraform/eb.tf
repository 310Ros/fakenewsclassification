terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-west-2"  
}

resource "aws_elastic_beanstalk_application" "my_app" {
  name = "fakenewsclfflaskapp"
}

resource "aws_elastic_beanstalk_environment" "my_environment" {
  name        = "fakenewsclfflaskappEnv"
  application = aws_elastic_beanstalk_application.my_app.name
  solution_stack_name = "64bit Amazon Linux 2023 v4.3.8 running Docker"
  
  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = "t2.small"
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "EnvironmentType"
    value     = "SingleInstance"
  }

   setting {
      namespace = "aws:autoscaling:launchconfiguration"
      name      = "IamInstanceProfile"
      value     = "aws-elasticbeanstalk-ec2-role"
    }
}

