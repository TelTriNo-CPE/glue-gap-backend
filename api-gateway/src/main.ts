import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as express from 'express';

const ONE_GB = 1024 * 1024 * 1024;
// 10 minutes — enough to upload a 1 GB file on a slow connection
const UPLOAD_TIMEOUT_MS = 10 * 60 * 1000;

async function bootstrap() {
  // bodyParser: false is required so multipart requests are not consumed
  // by Express's built-in body parser before Multer can stream them.
  const app = await NestFactory.create(AppModule, { bodyParser: false });

  // Manually register body parsers with a 1 GB limit so non-multipart
  // endpoints (e.g. JSON API calls) don't reject large payloads with 413.
  app.use(express.json({ limit: ONE_GB }));
  app.use(express.urlencoded({ limit: ONE_GB, extended: true }));

  // Allow any origin so browser-based frontends can upload directly.
  app.enableCors();

  const port = process.env.PORT ?? 3030;
  await app.listen(port);

  // Extend the underlying Node.js socket timeout so a 1 GB upload over a
  // slow connection isn't cut short by the default 5-second idle timeout.
  app.getHttpServer().setTimeout(UPLOAD_TIMEOUT_MS);

  console.log(`api-gateway listening on port ${port}`);
}
bootstrap();
