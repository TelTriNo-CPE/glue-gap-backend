import { Module } from '@nestjs/common';
import { CleanupCronService } from './cleanup.service';

@Module({
  providers: [CleanupCronService],
})
export class CleanupModule {}
