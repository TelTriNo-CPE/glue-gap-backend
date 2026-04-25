import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import {
  S3Client,
  HeadBucketCommand,
  HeadObjectCommand,
  CreateBucketCommand,
  DeleteObjectCommand,
} from '@aws-sdk/client-s3';

@Injectable()
export class UploadService implements OnModuleInit {
  private readonly logger = new Logger(UploadService.name);
  private readonly bucket = process.env.MINIO_BUCKET ?? 'glue-analysis';
  private readonly s3 = new S3Client({
    endpoint: process.env.MINIO_ENDPOINT,
    region: 'us-east-1',
    credentials: {
      accessKeyId: process.env.MINIO_ACCESS_KEY,
      secretAccessKey: process.env.MINIO_SECRET_KEY,
    },
    forcePathStyle: true,
  });

  async onModuleInit() {
    await this.ensureBucketExists();
  }

  private async ensureBucketExists() {
    try {
      await this.s3.send(new HeadBucketCommand({ Bucket: this.bucket }));
      this.logger.log(`Bucket "${this.bucket}" already exists`);
    } catch (err: any) {
      if (err?.name === 'NotFound' || err?.$metadata?.httpStatusCode === 404) {
        await this.s3.send(new CreateBucketCommand({ Bucket: this.bucket }));
        this.logger.log(`Bucket "${this.bucket}" created`);
      } else {
        this.logger.error(`Failed to verify bucket: ${err?.message}`);
        throw err;
      }
    }
  }

  async cancelUpload(key: string): Promise<void> {
    await this.s3.send(new DeleteObjectCommand({ Bucket: this.bucket, Key: key }));
  }

  async buildResponse(file: Express.MulterS3.File) {
    // multer-s3 relies on httpUploadProgress.total which is undefined for
    // S3 multipart uploads, so file.size is often 0 for large files.
    // Fall back to a HeadObject call to get the real size from MinIO.
    let size = file.size;
    if (!size && file.key) {
      try {
        const head = await this.s3.send(
          new HeadObjectCommand({ Bucket: this.bucket, Key: file.key }),
        );
        size = head.ContentLength ?? 0;
      } catch {
        size = 0;
      }
    }
    return {
      originalName: file.originalname,
      key: file.key,
      bucket: file.bucket,
      location: file.location,
      size,
      contentType: file.mimetype,
    };
  }
}
