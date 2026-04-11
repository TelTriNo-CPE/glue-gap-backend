import { Injectable, NotFoundException } from '@nestjs/common';
import {
  S3Client,
  GetObjectCommand,
  ListObjectsV2Command,
} from '@aws-sdk/client-s3';
import { Readable } from 'stream';

@Injectable()
export class TilesService {
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

  async getObject(minioKey: string): Promise<{ body: Readable; contentType: string }> {
    try {
      const result = await this.s3.send(
        new GetObjectCommand({ Bucket: this.bucket, Key: minioKey }),
      );
      const contentType = result.ContentType ?? this.inferContentType(minioKey);
      return { body: result.Body as Readable, contentType };
    } catch (err: any) {
      if (err?.name === 'NoSuchKey' || err?.$metadata?.httpStatusCode === 404) {
        throw new NotFoundException(`Tile not found: ${minioKey}`);
      }
      throw err;
    }
  }

  inferContentType(key: string): string {
    const lower = key.toLowerCase();
    if (lower.endsWith('.dzi')) return 'application/xml';
    if (lower.endsWith('.jpeg') || lower.endsWith('.jpg')) return 'image/jpeg';
    if (lower.endsWith('.png')) return 'image/png';
    return 'application/octet-stream';
  }

  async listTileSets(): Promise<string[]> {
    const result = await this.s3.send(
      new ListObjectsV2Command({
        Bucket: this.bucket,
        Prefix: 'tiles/',
        Delimiter: '/',
      }),
    );
    return (result.CommonPrefixes ?? [])
      .map((p) => p.Prefix?.replace(/^tiles\//, '').replace(/\/$/, '') ?? '')
      .filter(Boolean);
  }
}
