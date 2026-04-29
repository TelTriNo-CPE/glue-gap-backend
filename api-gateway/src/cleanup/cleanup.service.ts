import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import {
  S3Client,
  ListObjectsV2Command,
  DeleteObjectsCommand,
  ObjectIdentifier,
} from '@aws-sdk/client-s3';

const TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

@Injectable()
export class CleanupCronService {
  private readonly logger = new Logger(CleanupCronService.name);
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

  @Cron(CronExpression.EVERY_HOUR)
  async runTtlCleanup(): Promise<void> {
    this.logger.log('TTL Cleanup: Starting scan…');

    const cutoff = new Date(Date.now() - TTL_MS);
    const toDelete: ObjectIdentifier[] = [];

    let continuationToken: string | undefined;
    do {
      const res = await this.s3.send(
        new ListObjectsV2Command({
          Bucket: this.bucket,
          ContinuationToken: continuationToken,
        }),
      );

      for (const obj of res.Contents ?? []) {
        if (obj.Key && obj.LastModified && obj.LastModified < cutoff) {
          toDelete.push({ Key: obj.Key });
        }
      }

      continuationToken = res.NextContinuationToken;
    } while (continuationToken);

    if (toDelete.length === 0) {
      this.logger.log('TTL Cleanup: No old objects found.');
      return;
    }

    this.logger.log(`TTL Cleanup: Found ${toDelete.length} old object(s) to delete.`);

    // DeleteObjectsCommand accepts at most 1 000 keys per request.
    for (let i = 0; i < toDelete.length; i += 1000) {
      const batch = toDelete.slice(i, i + 1000);
      await this.s3.send(
        new DeleteObjectsCommand({
          Bucket: this.bucket,
          Delete: { Objects: batch, Quiet: true },
        }),
      );
    }

    this.logger.log(`TTL Cleanup: Successfully removed ${toDelete.length} old object(s).`);
  }
}
