import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UploadModule } from './upload/upload.module';
import { TilesModule } from './tiles/tiles.module';
import { ResultsModule } from './results/results.module';
import { ExportsModule } from './exports/exports.module';

@Module({
  imports: [UploadModule, TilesModule, ResultsModule, ExportsModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
