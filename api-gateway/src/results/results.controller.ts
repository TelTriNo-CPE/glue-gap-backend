import {
  BadRequestException,
  Body,
  Controller,
  Get,
  NotFoundException,
  Param,
  Post,
  Put,
  Res,
} from '@nestjs/common';
import { Response } from 'express';
import { ResultsService } from './results.service';

@Controller('results')
export class ResultsController {
  constructor(private readonly resultsService: ResultsService) {}

  @Post('analyze')
  async analyze(
    @Body() body: { key: string; sensitivity?: number; minArea?: number },
    @Res() res: Response,
  ) {
    try {
      const result = await this.resultsService.analyzeGaps(
        body.key,
        body.sensitivity,
        body.minArea,
      );
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.json(result);
    } catch (err: any) {
      const status = err?.status ?? 500;
      res.status(status).json({ message: err?.message ?? 'Internal error' });
    }
  }

  @Get(':stem')
  async getResult(@Param('stem') stem: string, @Res() res: Response) {
    let result: object;
    try {
      result = await this.resultsService.getResult(stem);
    } catch (err) {
      if (err instanceof NotFoundException) {
        res.status(404).json({ message: err.message });
        return;
      }
      throw err;
    }
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.json(result);
  }

  @Put(':stem/gaps')
  async updateGaps(
    @Param('stem') stem: string,
    @Body() body: { gaps: unknown[] },
    @Res() res: Response,
  ) {
    try {
      if (!Array.isArray(body?.gaps)) {
        throw new BadRequestException('Body must include a gaps array');
      }
      const result = await this.resultsService.updateResultGaps(stem, body.gaps);
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.json(result);
    } catch (err: any) {
      const status = err?.status ?? 500;
      res.status(status).json({ message: err?.message ?? 'Internal error' });
    }
  }
}
