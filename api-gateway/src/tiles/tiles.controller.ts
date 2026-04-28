import { Controller, Get, Head, NotFoundException, Req, Res } from '@nestjs/common';
import { Request, Response } from 'express';
import { TilesService } from './tiles.service';

@Controller('tiles')
export class TilesController {
  constructor(private readonly tilesService: TilesService) {}

  @Get()
  async listTileSets() {
    const stems = await this.tilesService.listTileSets();
    return { tiles: stems };
  }

  @Get('*')
  @Head('*')
  async proxy(@Req() req: Request, @Res() res: Response) {
    // req.path will be something like "/tiles/filename.dzi" or "/tiles/filename_files/..."
    const path = req.path.replace(/^\/tiles\//, '');
    let minioKey: string;

    if (path.endsWith('.dzi')) {
      const stem = path.replace('.dzi', '');
      minioKey = `tiles/${stem}/${stem}.dzi`;
    } else if (path.includes('_files/')) {
      const stem = path.split('_files/')[0];
      minioKey = `tiles/${stem}/${path}`;
    } else {
      minioKey = `tiles/${path}`;
    }

    let body: NodeJS.ReadableStream;
    let contentType: string;
    try {
      ({ body, contentType } = await this.tilesService.getObject(minioKey));
    } catch (err) {
      if (err instanceof NotFoundException) {
        res.status(404).json({ message: err.message });
        return;
      }
      throw err;
    }

    res.setHeader('Content-Type', contentType);
    res.setHeader('Access-Control-Allow-Origin', '*');
    body.on('error', () => res.destroy());
    body.pipe(res);
  }
}
