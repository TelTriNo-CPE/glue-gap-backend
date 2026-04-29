import { Module } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UploadModule } from './upload/upload.module';
import { TilesModule } from './tiles/tiles.module';
import { ResultsModule } from './results/results.module';
import { ExportsModule } from './exports/exports.module';
import { CleanupModule } from './cleanup/cleanup.module';

@Module({
  imports: [
    ScheduleModule.forRoot(),
    UploadModule,
    TilesModule,
    ResultsModule,
    ExportsModule,
    CleanupModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
