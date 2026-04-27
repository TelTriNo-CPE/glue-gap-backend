import { Injectable, NotFoundException } from '@nestjs/common';
import { GetObjectCommand, PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { Readable } from 'stream';

interface GapPayload {
  area_px: number;
  equiv_radius_px: number;
  centroid_norm: number[];
  coordinates: number[];
}

interface RadiusStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
}

interface AnalysisResultPayload {
  stem: string;
  image_size: { width: number; height: number };
  gap_count: number;
  gaps: GapPayload[];
  radius_stats: RadiusStats | null;
  sensitivity?: number | null;
  min_area?: number | null;
  [key: string]: unknown;
}

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

  async updateResultGaps(stem: string, gaps: unknown[]): Promise<object> {
    const current = await this.getResult(stem) as AnalysisResultPayload;
    const nextGaps = gaps as GapPayload[];

    const updated: AnalysisResultPayload = {
      ...current,
      stem,
      gaps: nextGaps,
      gap_count: nextGaps.length,
      radius_stats: this.calculateRadiusStats(nextGaps),
    };

    const body = JSON.stringify(updated);
    await this.s3.send(
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: `results/${stem}.json`,
        Body: body,
        ContentType: 'application/json',
      }),
    );

    return updated;
  }

  private calculateRadiusStats(gaps: GapPayload[]): RadiusStats | null {
    if (gaps.length === 0) {
      return null;
    }

    const radii = gaps
      .map((gap) => gap.equiv_radius_px)
      .sort((left, right) => left - right);
    const total = radii.reduce((sum, value) => sum + value, 0);
    const mean = total / radii.length;
    const middleIndex = Math.floor(radii.length / 2);
    const median = radii.length % 2 === 0
      ? (radii[middleIndex - 1] + radii[middleIndex]) / 2
      : radii[middleIndex];
    const variance = radii.reduce((sum, value) => sum + (value - mean) ** 2, 0) / radii.length;

    return {
      min: this.round(radii[0]),
      max: this.round(radii[radii.length - 1]),
      mean: this.round(mean),
      median: this.round(median),
      std: this.round(Math.sqrt(variance)),
    };
  }

  private round(value: number): number {
    return Math.round(value * 100) / 100;
  }
}
