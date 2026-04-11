import { Module } from '@nestjs/common';
import { MulterModule } from '@nestjs/platform-express';
import { S3Client } from '@aws-sdk/client-s3';
import multerS3 = require('multer-s3');
import { UploadController } from './upload.controller';
import { UploadService } from './upload.service';

const ONE_GB = 1024 * 1024 * 1024;

function buildS3Client(): S3Client {
  return new S3Client({
    endpoint: process.env.MINIO_ENDPOINT,
    region: 'us-east-1',
    credentials: {
      accessKeyId: process.env.MINIO_ACCESS_KEY,
      secretAccessKey: process.env.MINIO_SECRET_KEY,
    },
    forcePathStyle: true,
  });
}

@Module({
  imports: [
    MulterModule.register({
      storage: multerS3({
        s3: buildS3Client(),
        bucket: process.env.MINIO_BUCKET ?? 'glue-analysis',
        contentType: (_req, file, cb) => cb(null, file.mimetype),
        key(_req, file, cb) {
          const timestamp = Date.now();
          const random = Math.round(Math.random() * 1e9);
          const ext = file.originalname.split('.').pop();
          cb(null, `${timestamp}-${random}.${ext}`);
        },
      }),
      limits: { fileSize: ONE_GB },
    }),
  ],
  controllers: [UploadController],
  providers: [UploadService],
})
export class UploadModule {}
