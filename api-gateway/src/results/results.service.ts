import { Injectable, NotFoundException } from '@nestjs/common';
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { Readable } from 'stream';

@Injectable()
export class ResultsService {
  private readonly bucket = process.env.MINIO_BUCKET ?? 'glue-analysis';
  private readonly imageProcessorUrl =
    process.env.IMAGE_PROCESSOR_URL ?? 'http://image-processor:8080';
  private readonly s3 = new S3Client({
    endpoint: process.env.MINIO_ENDPOINT,
    region: 'us-east-1',
    credentials: {
      accessKeyId: process.env.MINIO_ACCESS_KEY,
      secretAccessKey: process.env.MINIO_SECRET_KEY,
    },
    forcePathStyle: true,
  });

  async analyzeGaps(
    key: string,
    sensitivity?: number,
    minArea?: number,
  ): Promise<object> {
    const response = await fetch(`${this.imageProcessorUrl}/analyze-gaps`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        key,
        sensitivity: sensitivity ?? 50,
        min_area: minArea ?? 20,
      }),
    });
    if (!response.ok) {
      const text = await response.text();
      const err: any = new Error(
        `image-processor error: ${response.status} ${text}`,
      );
      err.status = response.status;
      throw err;
    }
    return response.json();
  }

  async getResult(stem: string): Promise<object> {
    try {
      const res = await this.s3.send(
        new GetObjectCommand({ Bucket: this.bucket, Key: `results/${stem}.json` }),
      );
      const chunks: Buffer[] = [];
      for await (const chunk of res.Body as Readable) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      return JSON.parse(Buffer.concat(chunks).toString('utf-8'));
    } catch (err: any) {
      if (err?.name === 'NoSuchKey' || err?.$metadata?.httpStatusCode === 404) {
        throw new NotFoundException(`Result not found: ${stem}`);
      }
      throw err;
    }
  }
}
