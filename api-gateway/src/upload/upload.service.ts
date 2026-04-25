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

  async buildResponse(
    file: Express.MulterS3.File,
  ): Promise<{ originalName: string; key: string; bucket: string; location: string; size: number; contentType: string }> {
    // multer-s3 relies on httpUploadProgress.total which is undefined for
    // S3 multipart uploads, so file.size is often 0 for large files.
    // Always query MinIO for the authoritative object size.
    let size: number = file.size ?? 0;
    try {
      const head = await this.s3.send(
        new HeadObjectCommand({ Bucket: this.bucket, Key: file.key }),
      );
      const s3Size = head.ContentLength ?? 0;
      this.logger.log(
        `Upload "${file.key}": multer reported ${file.size ?? 0} B, MinIO reports ${s3Size} B`,
      );
      if (s3Size > 0) size = s3Size;
    } catch (err: any) {
      this.logger.warn(
        `HeadObject failed for "${file.key}": ${err?.message}`,
      );
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
