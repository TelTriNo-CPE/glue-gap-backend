import {
  Controller,
  Post,
  Delete,
  Param,
  UseInterceptors,
  UploadedFile,
  HttpCode,
  HttpStatus,
  BadRequestException,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { UploadService } from './upload.service';

@Controller('upload')
export class UploadController {
  constructor(private readonly uploadService: UploadService) {}

  @Post('image')
  @HttpCode(HttpStatus.CREATED)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 1024 * 1024 * 1024 } }))
  async uploadImage(@UploadedFile() file: Express.MulterS3.File) {
    if (!file) {
      throw new BadRequestException('No file provided. Use field name "file".');
    }
    const response = await this.uploadService.buildResponse(file);
    return response;
  }

  @Delete('cancel/:key')
  @HttpCode(HttpStatus.NO_CONTENT)
  async cancelUpload(@Param('key') key: string) {
    await this.uploadService.cancelUpload(key);
  }
}
