import { Controller, Get, NotFoundException, Param, Res } from '@nestjs/common';
import { Response } from 'express';
import { ExportsService } from './exports.service';

@Controller('exports')
export class ExportsController {
  constructor(private readonly exportsService: ExportsService) {}

  @Get(':stem/excel')
  async getExcel(@Param('stem') stem: string, @Res() res: Response) {
    let url: string;
    try {
      url = await this.exportsService.getExcelUrl(stem);
    } catch (err) {
      if (err instanceof NotFoundException) {
        res.status(404).json({ message: err.message });
        return;
      }
      throw err;
    }
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.json({ url });
  }

  @Get(':stem/image')
  async getImage(@Param('stem') stem: string, @Res() res: Response) {
    let url: string;
    try {
      url = await this.exportsService.getImageUrl(stem);
    } catch (err) {
      if (err instanceof NotFoundException) {
        res.status(404).json({ message: err.message });
        return;
      }
      throw err;
    }
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.json({ url });
  }
}
